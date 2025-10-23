from transformers import pipeline

model_name = "eryk-mazus/polka-1.1b"
generator = pipeline("text-generation", model=model_name, device=0)

print("Model loaded")

while True:
    prompt=None 
    while not prompt:
        prompt = input().strip()

    prompt += " to definicja słowa:"
    g = generator(
        prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=4, temperature=0.1
    )[0]["generated_text"]

    print(g)
    print(50 * "=")
    print()
    last_prompt = prompt
