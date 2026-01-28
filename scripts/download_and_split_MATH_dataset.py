from pathlib import Path

from datasets import load_dataset

root_dir = Path(__file__).parent.parent

dataset = load_dataset("qwedsacf/competition_math", split="train")
datasets = dataset.train_test_split(test_size=1000, shuffle=True, seed=42)
datasets.save_to_disk(root_dir.joinpath("data/a5-alignment/MATH"))
