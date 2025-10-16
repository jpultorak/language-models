import re
from transformers import pipeline, set_seed


def build_prompt(query: str, history=None) -> str:
    intro = (
        "SCENA: Rozmowa dwóch osob, na temat małp.\n\n"
        "Pytanie: Czy małpy lubią jeść patyki?\n"
        "Odpowiedź: Nie, małpy wolą jeść banany.\n"
    )

    lines = []
    if history:
        for u, b in history[-1:]:
            lines += [f"Pytanie: {u}", f"Odpowiedź: {b}"]
    lines += [f"Pytanie: {query}", "Odpowiedź: "]
    return "\n".join(lines)

def score(reply: str, user_query: str) -> float:
    s = len(reply)
    return s

def process_output(text: str) -> str:
    # text = text.split("\nUżytkownik:")[0].strip()

    last_sentence_idx = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
    if last_sentence_idx != -1:
        return text[:last_sentence_idx+1]
    return text

def main():
    generator = pipeline(
        'text-generation',
        model='flax-community/papuGaPT2',
        device=0
    )

    print("Model loaded")
    history = []

    while True:
        query = input("Pytanie: ").strip()
        if not query:
            continue

        prompt = build_prompt(query, history)

        outs = generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=2,
            return_full_text=False,
        )

        candidates = []
        for o in outs:
            text = o["generated_text"]
            processed = process_output(text)
            candidates.append(processed)

        # print("=======DEBUG:\n ", candidates," \nEND DEBUG=======\n")

        reply = min(candidates, key=lambda r: score(r, query))
        print(f"Odpowiedź: {reply}", "\n" + "="*50 + "\n")

        history.append((query, reply))
        # print("=======DEBUG HISTORY:\n ", history," \nEND DEBUG HISTORY=======\n")

if __name__ == "__main__":
    main()