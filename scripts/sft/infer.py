from pathlib import Path

import yaml
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from cs336_alignment.prompts import r1_zero_prompt
from cs336_alignment.sft.infer import generate_completion
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import numpy as np


curdir = Path(__file__).parent
with open(curdir / "config.yaml") as f:
    config = yaml.safe_load(f)

ds_val = load_from_disk("data/a5-alignment/MATH-SFT")["test"]
base_model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    dtype=config["model"]["dtype"],
    device_map=config["model"]["device_map"],
    attn_implementation=config["model"]["attn_implementation"],
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

model = PeftModel.from_pretrained(base_model, config["output"]["model_dir"])

indices = np.random.choice(len(ds_val), 3, replace=False)
for i in indices:
    sample = ds_val[i]
    problem: str = sample["problem"]  # type: ignore
    level: str = sample["level"]  # type: ignore
    type_: str = sample["type"]  # type: ignore
    solution: str = sample["solution"]  # type: ignore

    prompt = r1_zero_prompt(problem)
    completion = generate_completion(model, tokenizer, prompt, 4096, "</answer>")
    print(completion)

    rewards = r1_zero_reward_fn(completion, solution)
    print(rewards)
