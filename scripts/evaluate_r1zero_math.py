from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import evaluate_vllm
from cs336_alignment.prompts import r1_zero_prompt

root_dir = Path(__file__).parent.parent

model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
sampling_params = SamplingParams(
    max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
)

dataset = load_dataset(root_dir.joinpath("data/a5-alignment/MATH").as_posix())["test"]
prompts: list[str] = []
solutions: list[str] = []

for sample in dataset:
    solutions.append(sample["solution"])  # type: ignore
    prompt = r1_zero_prompt(sample["problem"])  # type: ignore
    prompts.append(prompt)


def callback(**kwargs):
    print(kwargs)


rewards = evaluate_vllm(
    model, r1_zero_reward_fn, prompts, solutions, sampling_params, callback
)

# (1) correct with both format and answer reward 1,
# (2) format reward 1 and answer reward 0,
# (3) format reward 0 and answer reward 0?
# Observing at least 10 cases where format reward is 0,
# do you think the issue is with the base model’s output, or the parser?
# Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?

case1 = 0
case2 = 0
case3 = 0

for reward in rewards:
    format_flag = reward["format_reward"] == 1.0
    answer_flag = reward["answer_reward"] == 1.0

    if format_flag and answer_flag:
        case1 += 1
    elif format_flag and not answer_flag:
        case2 += 1
    elif not format_flag and answer_flag:
        case3 += 1

print("Total Summary:")
print(f"(1) correct with both format and answer reward 1: {case1}/{len(prompts)}")
print(f"(2) format reward 1 and answer reward 0: {case2}/{len(prompts)}")
print(f"(3) format reward 0 and answer reward 1: {case3}/{len(prompts)}")
