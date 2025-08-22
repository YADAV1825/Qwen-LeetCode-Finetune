import json
from tqdm import tqdm

input_file = "./dataset/LeetCodeDataset.jsonl"
output_file = "./dataset/train.json"

formatted_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing dataset"):
        data = json.loads(line)
        
        # Extract fields safely
        problem = data.get("problem_description", "")
        starter_code = data.get("starter_code", "")
        difficulty = data.get("difficulty", "")
        tags = data.get("tags", [])
        solution = data.get("completion", "")
        explanation = data.get("response", "")

        # Skip incomplete entries
        if not problem.strip() or not solution.strip():
            continue

        # Create instruction
        instruction = f"Solve this DSA problem in C++ and explain step by step:\n{problem}"
        if starter_code:
            instruction += f"\n\nStarter Code:\n{starter_code}"
        if difficulty:
            instruction += f"\n\nDifficulty: {difficulty}"
        if tags:
            instruction += f"\nTags: {', '.join(tags)}"

        # Combine code + reasoning
        output = solution
        if explanation.strip():
            output += f"\n\nExplanation:\n{explanation}"

        formatted_data.append({
            "instruction": instruction,
            "output": output
        })

# Save JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(formatted_data)} samples to {output_file}")
