from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from cs336_alignment.prompts import r1_zero_prompt
from cs336_alignment.sft.infer import generate_completion
import yaml

curdir = Path(__file__).parent
with open(curdir / "config.yaml") as f:
    config = yaml.safe_load(f)

ds_val = load_from_disk("data/a5-alignment/MATH-SFT")["test"]
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    dtype=config["model"]["dtype"],
    device_map=config["model"]["device_map"],
    attn_implementation=config["model"]["attn_implementation"],
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])


for sample in ds_val:
    problem: str = sample["problem"]  # type: ignore
    level: str = sample["level"]  # type: ignore
    type_: str = sample["type"]  # type: ignore
    solution: str = sample["solution"]  # type: ignore

    prompt = r1_zero_prompt(problem)
    completion = generate_completion(model, tokenizer, prompt, 512, "</answer>")
    print(completion)

    break
