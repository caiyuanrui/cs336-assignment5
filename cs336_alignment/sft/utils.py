from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import (
    PreTrainedModel,  # pyright: ignore[reportPrivateImportUsage]
    PreTrainedTokenizerBase,  # pyright: ignore[reportPrivateImportUsage]
)


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """

    assert len(prompt_strs) == len(output_strs)

    batch_size = len(prompt_strs)

    prompt_ids = [tokenizer.encode(s) for s in prompt_strs]
    output_ids = [tokenizer.encode(s) for s in output_strs]

    max_len = max(len(prompt_ids[i]) + len(output_ids[i]) for i in range(batch_size))

    # prompt_and_output_tensor
    pao_tensor = torch.empty(batch_size, max_len, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.bool)

    for i in range(batch_size):
        p = prompt_ids[i]
        o = output_ids[i]
        concat = p + o

        pao_tensor[i][: len(concat)] = torch.tensor(
            concat, dtype=torch.long, device=pao_tensor.device
        )
        response_mask[i][len(p) - 1 : len(concat) - 1] = True

    input_ids = pao_tensor[:][:-1]
    labels = pao_tensor[:][1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(
    logits: Float[Tensor, "batch_size seq_len vocab_size"],  # noqa: F722
) -> Float[Tensor, "batch_size seq_len"]:  # noqa: F722
    """
    Computes the per-token entropy of next-token predictions.
    Args:
        logits: (batch_size, seq_len, vocab_size)
    Returns:
        (batch_size, seq_len)
    """
    logsumexp: Float[Tensor, "batch_size seq_len"] = torch.logsumexp(logits, dim=-1)  # noqa: F722
    probs: Float[Tensor, "batch_size seq_len"] = torch.softmax(logits, -1)  # noqa: F722
    out = logsumexp - torch.sum(logits * probs, dim=-1)
    return out


@torch.inference_mode()
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    logits: Float[Tensor, "batch_size seq_len vocab_size"] = model(input_ids).logits  # type: ignore  # noqa: F722
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token_log_prob = torch.gather(log_probs, -1, labels[..., None]).squeeze(-1)

    result = {"log_probs": per_token_log_prob}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """
    Sums over tensor elements and normalizes by a constant while respecting a boolean mask.
    Args:
        - tensor: the tensor to sum and normalize
        - mask: same shape as `tensor`; positions with 1 are included in the sum
        - normalize_constant: the constant to divide by for normalization
        - dim: the dimension to sum along before normalization, if `None`, sum all dimensions
    Returns
        the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """
    return torch.sum(tensor * mask.to(tensor.dtype), dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Float[Tensor, "batch_size seq_len"],  # noqa: F722
    response_mask: Int[Tensor, "batch_size seq_len"],  # noqa: F722
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
):
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        - policy_log_probs: per-token log-probabilities from the SFT policy being trained
        - response_mask: 1 for response tokens, 0 for prompt/padding
        - gradient_accumulation_steps: number of microbatches per optimizer step
        - normalize_constant: the constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        - loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        - metadata: dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    loss = masked_normalize(
        -policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant * gradient_accumulation_steps,
    )
    loss.backward()

    loss = loss.detach()

    metadata: dict[str, Any] = {
        "loss": loss,
        "num_tokens": response_mask.sum().detach(),
    }
    return loss, metadata


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM  # pyright: ignore[reportPrivateImportUsage]

    model_name = "Qwen/Qwen3-0.6B-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    print("Prompt:", text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip(
        "\n"
    )

    print("thinking content:", thinking_content)
    print("content:", content)
