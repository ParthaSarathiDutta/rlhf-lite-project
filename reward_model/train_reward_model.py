import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Convert to ranking format: (better, worse)
    rows = []
    for _, row in df.iterrows():
        if row["preference"] == "A":
            rows.append({"text": row["prompt"] + " " + row["response_a"], "label": 1})
            rows.append({"text": row["prompt"] + " " + row["response_b"], "label": 0})
        else:
            rows.append({"text": row["prompt"] + " " + row["response_a"], "label": 0})
            rows.append({"text": row["prompt"] + " " + row["response_b"], "label": 1})
    return Dataset.from_pandas(pd.DataFrame(rows))

def tokenize(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_data("data/sample_preferences.csv")
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    dataset = dataset.train_test_split(test_size=0.2)

    args = TrainingArguments(
        output_dir="reward_model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("reward_model_output")
    tokenizer.save_pretrained("reward_model_output")

if __name__ == "__main__":
    main()
