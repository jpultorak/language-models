from dataclasses import dataclass
from pathlib import Path
from transformers import pipeline

from lab1.sentence_probability import sentence_prob


@dataclass(frozen=True)
class Review:
    positive: bool
    content: str


def read_reviews(path: Path) -> list[Review]:
    reviews = []
    with path.open() as f:
        for line in f.readlines():
            r_type, content = line.split(maxsplit=1)

            if r_type == "GOOD":
                positive = True
            elif r_type == "BAD":
                positive = False
            else:
                raise RuntimeError("Unknown review type")

            reviews.append(Review(positive=positive, content=content))

    return reviews

def sample_reviews():
    r1 = Review(
        positive=True,
        content="Wspaniała obsługa, jestem zachwycony, 10/10!"
    )

    r2 = Review(
        positive=False,
        content="Było okropnie, nie polecam nikomu usług tego Pana!"
    )
    return [r1, r2]

def load_model(model_name = "eryk-mazus/polka-1.1b"):
    generator = pipeline("text-generation", model=model_name, device=0)
    print("Model loaded")
    return generator

def is_positive_review(r: Review):
    good = f"Pozytywna opinia: {r.content}"
    bad = f"Negatywna opinia: {r.content}"

    return sentence_prob(good) > sentence_prob(bad)

def main():

    # generator = load_model()
    repo_root = Path(__file__).resolve().parents[2]
    reviews  = read_reviews(repo_root / "datasets" / "reviews_for_task3.txt")
    # reviews = sample_reviews()

    res = 0
    total = len(reviews)
    for r in reviews:
        ok = (is_positive_review(r) == r.positive)
        res  += ok
        print(f"POSITIVE: {r.positive}; GOOD ANS: {ok} \n{r.content}")

    print(res, total, res/total)





if __name__ == "__main__":
    main()