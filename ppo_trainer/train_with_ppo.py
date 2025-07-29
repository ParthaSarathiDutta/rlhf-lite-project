import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset

# Load policy model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO config
config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=4,
    ppo_epochs=4,
    log_with=None,
)

# Simulated prompts
prompts = ["What is RLHF?", "Explain PPO.", "What is a language model?"]
tokenized_prompts = [tokenizer(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]

# Dummy reward function for demonstration
def dummy_reward_fn(query, response):
    return torch.tensor([1.0]) if "align" in response else torch.tensor([0.1])

# PPO trainer
ppo_trainer = PPOTrainer(config, model, tokenizer)

# Training loop
for epoch in range(3):
    for query_input in tokenized_prompts:
        query_input = query_input.unsqueeze(0)
        response_tensor = model.generate(query_input, max_new_tokens=20)
        response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
        reward = dummy_reward_fn(query_input, response)

        # Run PPO step
        ppo_trainer.step([query_input[0]], [response_tensor[0]], rewards=[reward])
        print(f"Epoch {epoch}: Prompt: {tokenizer.decode(query_input[0])} → {response} | Reward: {reward.item()}")
