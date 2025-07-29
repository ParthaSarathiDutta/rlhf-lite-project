# RLHF-Toy-Language-Model

This project demonstrates a minimal working pipeline for **Reinforcement Learning from Human Feedback (RLHF)** using Hugging Face's `trl` library.

---

## 🔍 What is RLHF?

RLHF is a framework for fine-tuning language models using human preferences, enabling alignment with desirable behaviors. It's central to the training of models like ChatGPT.

---

## 🛠️ Project Structure

- `data/`: Simulated human preferences
- `reward_model/`: Train reward model to distinguish preferred responses
- `ppo_trainer/`: PPO loop to align LLM using reward model
- `eval/`: Score generated responses with reward model
- `utils/`: Logging utilities
- `config.json`: Model and training config

---

## Example Training Flow

1. Train a reward model:
```bash
python reward_model/train_reward_model.py
```

2. Fine-tune a GPT2 model with PPO:
```bash
python ppo_trainer/train_with_ppo.py
```

3. Evaluate alignment:
```bash
python eval/evaluate_model.py
```

---

## Key Technologies

- `transformers`, `trl`, `torch`, `datasets`, `accelerate`
- Reward modeling via classification
- PPO training loop

---

## Why This Project?

This is a compact yet powerful demonstration of:
- Human-in-the-loop learning
- Reward modeling
- LLM fine-tuning with reinforcement learning


---

## License

MIT
