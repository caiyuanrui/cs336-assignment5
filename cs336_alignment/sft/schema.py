from dataclasses import dataclass


@dataclass
class MathSftRow:
    id: int
    question: str
    reasoning: str
    answer: str
