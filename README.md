# Qwen-LeetCode-Finetune

# Fine-Tuning Qwen 2.5 Coder for LeetCode Problem Solving

## 📌 Overview
This project fine-tunes the **Qwen 2.5 Coder** model on a dataset of **5,000 LeetCode problems** to improve its ability to solve algorithmic coding challenges.  
Using **Low-Rank Adaptation (LoRA)**, the model was specialized to generate optimized, accurate, and executable solutions for competitive programming problems.

---

## 🎯 Objectives
- Adapt Qwen 2.5 Coder for **domain-specific problem solving**.  
- Improve solution accuracy, edge-case handling, and code optimization.  
- Efficiently fine-tune without excessive computational overhead.  

---

## ⚙️ Methods & Tools

### Environment
- **Hardware:** NVIDIA GeForce RTX 3070 Ti Laptop GPU  
- **Libraries:**  
  - `torch` – Model training & GPU acceleration  
  - `transformers` – Model/tokenizer handling  
  - `peft` – LoRA fine-tuning  
  - `datasets` – Dataset management  
  - `trl` – Supervised Fine-Tuning (SFTTrainer)  

### Model Selection
- **Base Model:** Qwen 2.5 Coder  
- **Adaptation Method:** LoRA (Low-Rank Adaptation)  
  - Rank (r): 16  
  - Alpha: 32  
  - Target Modules: `q_proj`, `v_proj`  
  - Dropout: 0.05  

### Dataset
- **Size:** 5,000 LeetCode problems  
- **Format:** JSON with fields:  

---

```json
{
  "instruction": "Given an array of integers, find two numbers that add up to a specific target.",
  "output": "def two_sum(nums, target):\n  hash_map = {}\n  for i, num in enumerate(nums):\n    complement = target - num\n    if complement in hash_map:\n      return [hash_map[complement], i]\n    hash_map[num] = i"
}
```

---

#🛠️ Fine-Tuning Setup
Epochs: 60

Batch Size: 1 (GPU memory constrained)

Learning Rate: 2e-4 (linear decay)

Gradient Accumulation: 1

Sequence Length: 1024 tokens

Precision: FP16 enabled

Data was formatted into prompt-response pairs:

python
Copy
Edit
def formatting_func(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return [prompt]
📊 Training Results
Sample logs:

yaml
Copy
Edit
Epoch 0.01: {'loss': 0.5775, 'mean_token_accuracy': 0.8757}
Epoch 0.10: {'loss': 0.5757, 'mean_token_accuracy': 0.8745}
Epoch 3.00: {'loss': 0.5901, 'mean_token_accuracy': 0.8781}
Final Accuracy: ~87.8% mean token accuracy

---

Output Directory: ./output/lora-qwen-leetcode

🚀 Example Output
Problem
Instruction:
Find two numbers in an array that add up to a target.

Model Output (after fine-tuning)
python
Copy
Edit
def two_sum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
The fine-tuned model generates clean, optimized, and executable code compared to baseline responses.

✅ Conclusion
Fine-tuned Qwen 2.5 Coder with LoRA for LeetCode problem solving.

Achieved 87.8% token accuracy with optimized solution generation.

Demonstrated feasibility of adapting large models to domain-specific coding tasks.

---

🔮 Future Work
Expand dataset with problems from HackerRank, Codeforces, AtCoder.

Explore larger models and configurations.

Develop a real-time web interface for interactive problem solving.

📚 References
LoRA: Low-Rank Adaptation

Qwen Models
