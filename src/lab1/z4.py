from dataclasses import dataclass
from pathlib import Path
from transformers import pipeline


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


def answer_question(q: Question, generator):

    prompt = f"Odpowiedź na pytanie: {q.question} to: "
    g = generator(
        prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=15
    )[0]["generated_text"]

    return g

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    z4_path = repo_root / "datasets" / "p1" / "z4"
    qs = read_questions(z4_path)

    generator = load_model()

    for q in qs[:5]:
        print(f"{q}LM answer: {answer_question(q, generator)}")
        print("-----------------------------------")




