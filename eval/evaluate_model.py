import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate(prompt, response):
    model_path = "reward_model_output"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    text = prompt + " " + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    score = torch.softmax(outputs.logits, dim=-1)[0][1].item()  # Prob of 'good' class
    return score

if __name__ == "__main__":
    prompt = "What is RLHF?"
    response = "RLHF is a method that aligns language models with human feedback."
    score = evaluate(prompt, response)
    print(f"Reward score: {score:.4f}")
