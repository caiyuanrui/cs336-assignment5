# We have to make our own SFT dataset through Deepseek Reasoning Model.
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import json
from tqdm import tqdm

from cs336_alignment.data import get_MATH_dataset


def get_start_index(path: Path) -> int:
    if not path.exists():
        return 0

    last_line = None

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                last_line = line

    if last_line is None:
        return 0

    try:
        rec = json.loads(last_line)
        last_id = int(rec["id"])
        return last_id + 1
    except Exception as e:
        print(f"[WARN] Failed to parse last line of {path}: {e}")
        print(
            "[WARN] Will start from 0. If this is unexpected, please inspect the file manually."
        )
        return 0


load_dotenv()

root_dir = Path(__file__).parent.parent.parent
output_path = root_dir / "data/MATH/sft.jsonl"
dataset = get_MATH_dataset()


def main():
    n = len(dataset)
    start_idx = get_start_index(output_path)

    if start_idx >= n:
        print(
            f"[INFO] sft.jsonl already completed (start_idx={start_idx}, len={n}). Nothing to do."
        )
        return

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )

    mode = "a" if start_idx > 0 else "w"
    print(
        f"[INFO] Writing to {output_path} with mode='{mode}', starting from id={start_idx}/{n}"
    )

    with open(output_path, mode) as f:
        for i in tqdm(
            range(start_idx, n),
            total=n - start_idx,
            initial=start_idx,
            desc="Generating SFT data",
        ):
            sample = dataset[i]
            question: str = sample["problem"]  # type: ignore

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                stream=False,
                max_tokens=2048,
            )

            reasoning_content: str = response.choices[0].message.reasoning_content  # type: ignore
            answer_content = response.choices[0].message.content

            record = {
                "id": i,
                "question": question,
                "reasoning": reasoning_content,
                "answer": answer_content,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
