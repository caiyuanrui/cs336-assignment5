from functools import lru_cache
from pathlib import Path

_prompts_dir = Path(__file__).parent.joinpath("prompts")


@lru_cache(maxsize=1)
def r1_zero_template():
    with open(_prompts_dir.joinpath("r1_zero.prompt")) as f:
        return f.read()


def r1_zero_prompt(question: str):
    return r1_zero_template().format(question=question)
