from transformers import pipeline

generator = pipeline("text-generation", model="flax-community/papuGaPT2", device=0)

print("Model loaded")

while True:
    prompt=None 
    while not prompt:
        prompt = "Definicja słowa. " +input().strip()

    g = generator(
        prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=4, temperature=0.1
    )[0]["generated_text"]

    print(g)
    print(50 * "=")
    print()
    last_prompt = prompt
