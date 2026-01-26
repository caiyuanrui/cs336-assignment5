import numpy as np
from datasets import Dataset, load_dataset


def get_MATH_dataset():
    return load_dataset("qwedsacf/competition_math", split="train")


def sample_MATH_as_valid(
    *, dataset: Dataset | None = None, size=100, seed=42
) -> Dataset:
    if dataset is None:
        dataset = get_MATH_dataset()  # type: ignore
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size, replace=False)
    return dataset.select(indices)
