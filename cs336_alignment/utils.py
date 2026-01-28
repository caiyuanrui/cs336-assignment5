import torch
from jaxtyping import Float
from torch import Tensor
from transformers import (
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
