from typing import Callable

from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
    callback: Callable | None = None,
):
    outputs = vllm_model.generate(prompts, sampling_params)
    rewards: list[dict[str, float]] = []

    for output, gt in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward = reward_fn(response, gt)
        rewards.append(reward)
        if callback is not None:
            callback(response=response, reward=reward)

    return rewards
