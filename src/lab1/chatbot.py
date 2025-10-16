import re
from transformers import pipeline, set_seed


def build_prompt(query, history):
    header = "SCENA: krótka rozmowa. Odpowiadaj zwięźle i nie powtarzaj się.\n\n"
    lines = [header]
    for u, b in history[-1:]:
        lines += [f"Pytanie: {u}", f"Odpowiedź: {b}"]
    lines += [f"Pytanie: {query}", "Odpowiedź:"]
    return "\n".join(lines)

def score(reply: str, user_query: str) -> float:
    s = 0

    words = reply.split()

    consecutive_words = list(zip(words, words[1:]))
    repeats = len(consecutive_words ) - len(set(consecutive_words))
    s += 3 * repeats

    runs = re.findall(r"\b(\w+)(?:\s+\1\b)+", reply.lower())
    s += 9 * len(runs)

    p_cnt = words.count("pytanie:") + words.count("odpoweidź:")
    s += 100 * min(p_cnt, 1)
    return s


def process_output(text: str) -> str:
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
            min_new_tokens=10,
            max_new_tokens=40,
            num_return_sequences=10,
            return_full_text=False,
            do_sample=True,
            top_p=0.9, temperature=0.8,
        )

        candidates = [process_output(o["generated_text"]) for o in outs]
        print("=======DEBUG:\n ", candidates," \nEND DEBUG=======\n")

        reply = min(candidates, key=lambda r: score(r, query))
        print(f"Odpowiedź: {reply}", "\n" + "="*50 + "\n")

        history.append((query, reply))
        # print("=======DEBUG HISTORY:\n ", history," \nEND DEBUG HISTORY=======\n")

if __name__ == "__main__":
    main()