import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = stop_ids
        self.L = len(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # input_ids: (batch, seq_len)
        if input_ids.shape[1] < self.L:
            return False

        tail = input_ids[0, -self.L :].tolist()
        return tail == self.stop_ids


@torch.inference_mode()
def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int, eot: str):
    stop_ids = tokenizer.encode(eot, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StopOnSubsequence(stop_ids)])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return completion
