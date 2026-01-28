from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    return LLM(
        model=model_id,
        device=device,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def load_policy_into_vllm_instance(): ...
