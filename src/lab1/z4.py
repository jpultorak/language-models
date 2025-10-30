import math
from dataclasses import dataclass
from pathlib import Path
from transformers import pipeline
import re
from lab1.sentence_probability import sentence_prob


@dataclass(frozen=True)
class Question:
    question: str
    answer: str

    def __str__(self):
        return f"Q: {self.question}\nA: {self.answer}"


def read_questions(path: Path) -> list[Question]:

    q_path = path / "task4_questions.txt"
    a_path = path / "task4_answers.txt"
    questions = []

    with q_path.open() as q_file, a_path.open() as a_file:
        for q, a in zip(q_file, a_file):
            questions.append(Question(question=q.rstrip(), answer=a.rstrip()))

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
        repetition_penalty=1.5,
    )[0]["generated_text"]

    return g.strip()


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

    prompt = f"Pytanie: {q}\nOdpowiedz krótko (1-3 słowa).\nOdpowiedź:"
    return gen(prompt, generator)

def answer_yes_no(q: str):
    prompt = f"Pytanie: {q}\nOdpowiedz jednym słowem (tak/nie).\nOdpowiedź:"

    s_yes = sentence_prob(prompt + " tak")
    s_no = sentence_prob(prompt + " nie")

    return "tak" if s_yes >= s_no else "nie"

def answer_x_y(q: str, x: str, y: str):
    prompt = f"Pytanie: {q}\nOdpowiedz jednym słowem f({x}/{y}).\nOdpowiedź:"

    s_x = sentence_prob(prompt + f" {x}")
    s_y = sentence_prob(prompt + f" {y}")

    return x if s_x >= s_y else y

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    z4_path = repo_root / "datasets" / "p1" / "z4"
    qs = read_questions(z4_path)

    generator = load_model()

    for q in qs[330:360]:
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

        last_czy = -1
        for (i, w) in enumerate(words):
            if w == "czy":
                last_czy = i

        if  last_czy != -1:
            if last_czy == 0:
                print("YES or NO QUESTION")
                answer = answer_yes_no(q.question)
            else:
                x = words[last_czy-1]
                y = words[last_czy+1]
                if y[-1] == '?':
                    y = y[:-1]
                print(f"{x} or {y} QUESTION!")
                answer = answer_x_y(q.question, x, y)

        if answer is None:
            answer = generic_answer(q.question, generator)

        print(f"{q}\nLM answer: {answer}")
        print("====================================")
        print()


