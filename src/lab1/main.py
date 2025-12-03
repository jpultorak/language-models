from transformers import pipeline

model_name = "eryk-mazus/polka-1.1b"
generator = pipeline("text-generation", model=model_name, device=0)

print("Model loaded")

while True:
    inp=None
    while not inp:
        inp = input().strip()

    prompt = "Twoim zadaniem jest przetłumaczyć zdania z Angielskiego na Polski:\nI like dogs. - Lubię psy.\nI am John. - Nazywam się Janek.\n" + inp
    g = generator(
        prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=15
    )[0]["generated_text"]

    print(g)
    print(50 * "=")
    print()
    last_prompt = prompt
