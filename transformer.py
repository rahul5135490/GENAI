from transformers import pipeline

def generate_text(prompt, max_length=50):
    generator = pipeline("text-generation", model="gpt2", framework="pt")  # "pt" = PyTorch
    results = generator(prompt, max_length=max_length, num_return_sequences=1)
    return results[0]['generated_text']

output = generate_text("I love you")
print(output)