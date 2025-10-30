from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    content: str
    answer: str



def read_questions(path: Path) -> list[Question]:

    q_path = path / "task4_questions.txt"
    a_path = path / "task4_answers.txt"
    questions = []

    with q_path.open() as q_file, a_path.open() as a_file:
        for q, a in zip(q_file, a_file):
            questions.append(Question(content=q, answer=a))

    return questions


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    z4_path = repo_root / "datasets" / "p1" / "z4"
    qs = read_questions(z4_path)
    print(qs[:5])
