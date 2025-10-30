import math
from dataclasses import dataclass
from pathlib import Path
from transformers import pipeline
import re

@dataclass(frozen=True)
class Question:
    question: str
    answer: str

    def __str__(self):
        return f"Q: {self.question}A: {self.answer}"


def read_questions(path: Path) -> list[Question]:

    q_path = path / "task4_questions.txt"
    a_path = path / "task4_answers.txt"
    questions = []

    with q_path.open() as q_file, a_path.open() as a_file:
        for q, a in zip(q_file, a_file):
            questions.append(Question(question=q, answer=a))

    return questions

def load_model(model_name = "eryk-mazus/polka-1.1b"):
    generator = pipeline("text-generation", model=model_name, device=0)
    print("Model loaded")
    return generator


def gen(prompt: str, generator, max_tokens = 10) -> str:
    g = generator(
        prompt,
        do_sample=False,
        max_new_tokens=max_tokens,
        return_full_text=False,
    )[0]["generated_text"]

    return g


def answer_year(q: str, generator) -> int | None:
    prompt = (
        "Podaj jeden rok związany z pytaniem. Tylko cyfry.\n"
        f"Pytanie: {q}\nRok:"
    )
    g = gen(prompt, generator, max_tokens=10)
    m = re.search(r"\d+", g)
    return int(m.group()) if m else 2137

def answer_century(q: str, generator) -> int | None:
    year = answer_year(q, generator)
    if year != 2137 and year >= 0:
        return math.ceil(year/100)
    return 2137

def generic_answer(q: str, generator):
    prompt = f"Odpowiedź na pytanie: {q} to: "
    return gen(prompt, generator)

# def answer_yes_no(q: Question, generator):
#     prompt = f"Odpowiedz jednym słowem (tak/nie).\nPytanie: {q.question}\nOdpowiedź:"
#     g = generator(
#         prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=10, return_full_text=False,
#     )[0]["generated_text"]
#
#     return g

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    z4_path = repo_root / "datasets" / "p1" / "z4"
    qs = read_questions(z4_path)

    generator = load_model()

    for q in qs[100:400]:
        words = q.question.lower().split()
        answer = None
        if words[0:2] == ["w", "którym"] and words[2] in ("roku", "wieku"):
            w = words[2]
            if w == "roku":
                print("PYTANIE O ROK!")
                answer = answer_year(q.question, generator)
            else:
                print("PYTANIE O WIEK!")
                answer = answer_century(q.question, generator)

        if answer is not None:
            print(f"{q}LM answer: {answer}")
            print("====================================")

        # if answer is None:
        #     answer = generic_answer(q.question, generator)
        #
        # print(f"{q}LM answer: {answer}")
        # print("====================================")



