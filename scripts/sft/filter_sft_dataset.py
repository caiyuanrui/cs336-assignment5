import json
from datasets import load_dataset

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.prompts import r1_zero_prompt

dec = json.decoder.JSONDecoder()
dataset = load_dataset("qwedsacf/competition_math", split="train")

# prompt: str
# response: str
# solution: str
positive_samples: list[dict[str, str]] = []
negative_samples: list[dict[str, str]] = []


with open("data/a5-alignment/MATH-SFT.jsonl") as f:
    for i in range(len(dataset)):
        line = f.readline()
        if line == "":
            break
        line_dec = dec.decode(line.strip())

        question: str = line_dec["question"]
        reasoning: str = line_dec["reasoning"]
        answer: str = line_dec["answer"]

        if reasoning == "" or answer == "":
            continue

        prompt = r1_zero_prompt(question)
        response = rf"{reasoning}</think> <answer>{answer}</answer>"
        solution = dataset[i]["solution"]

        rewards = r1_zero_reward_fn(response, solution)

        if rewards["reward"] > 0:
            positive_samples.append(
                {"prompt": prompt, "response": response, "solution": solution}
            )
        else:
            negative_samples.append(
                {"prompt": prompt, "response": response, "solution": solution}
            )

with open("data/a5-alignment/MATH/sft.jsonl", "x") as f:
    f.writelines([json.dumps(line) + "\n" for line in positive_samples])
