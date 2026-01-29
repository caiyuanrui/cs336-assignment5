import json
from datasets import load_dataset, Dataset, DatasetDict

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.prompts import r1_zero_prompt

dec = json.decoder.JSONDecoder()
dataset = load_dataset("qwedsacf/competition_math", split="train")

# prompt: str
# response: str
# solution: str
positive_samples: list[dict[str, str]] = []
negative_samples: list[dict[str, str]] = []

train_MATH_dataset_ids: list[int] = []

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
            train_MATH_dataset_ids.append(i)
        else:
            negative_samples.append(
                {"prompt": prompt, "response": response, "solution": solution}
            )

# generate sft dataset
try:
    with open("data/a5-alignment/MATH/sft.jsonl", "x") as f:
        f.writelines([json.dumps(line) + "\n" for line in positive_samples])
except Exception as e:
    print(e)

# split MATH dataset
train_dataset = Dataset.from_list([dataset[i] for i in train_MATH_dataset_ids])
test_dataset = Dataset.from_list(
    [dataset[i] for i in range(len(dataset)) if i not in train_MATH_dataset_ids]
)

dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset_dict.save_to_disk("data/a5-alignment/MATH-SFT")
