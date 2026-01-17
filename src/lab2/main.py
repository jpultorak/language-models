from transformers import pipeline

model_name = "eryk-mazus/polka-1.1b"
generator = pipeline("text-generation", model=model_name, device=0)

print("Model loaded")


for a in range(0, 10):
    for b in range(0, 10):
        prompt = f"{a} + {b} ="
        g = generator(
            prompt, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=15
        )[0]["generated_text"]

        print(g)
        print(50 * "=")
        print()
