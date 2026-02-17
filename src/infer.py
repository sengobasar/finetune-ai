# src/infer.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "model/finetuned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print("\nLogicBot v2 is ready. Type 'exit' to quit.\n")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nBot:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    response = decoded.split("Bot:")[-1].strip()

    print("\nBot:", response)
    print()
