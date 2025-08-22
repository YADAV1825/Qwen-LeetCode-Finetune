import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------------- SETTINGS ----------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATA_PATH = "./dataset/cleaned_dataset.json"
OUTPUT_DIR = "./output/lora-qwen2.5-leetcode"

EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
SEQ_LENGTH = 1024
LOG_STEPS = 10

# ---------------- DEVICE CHECK ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "❌ CUDA GPU is required for 4-bit training."
print(f"✅ Using device: {torch.cuda.get_device_name(0)}")

# ---------------- LOAD TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ---------------- LOAD MODEL IN 4-BIT ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# ---------------- APPLY LoRA ----------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ---------------- LOAD DATA ----------------
dataset = load_dataset("json", data_files=DATA_PATH)

# ---------------- FORMAT EXAMPLES ----------------
def formatting_func(example):
    description = example.get("description", "").strip()
    solution = example.get("solution", "").strip()

    # Safety check
    if not description or not solution:
        return ""

    messages = [
        {"role": "user", "content": description},
        {"role": "assistant", "content": solution}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# ---------------- TRAINING ARGS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOG_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
    gradient_checkpointing=True
)

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    formatting_func=formatting_func
)

# ---------------- TRAIN ----------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"✅ Training complete. LoRA saved to: {OUTPUT_DIR}")

# PS C:\LLM> python -u "c:\LLM\scripts\train_lora_safe.py"
# ✅ Using device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
# Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.72s/it]
# trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193
# Applying formatting function to train dataset: 100%|████████████████████████████████████████████████████████████| 3363/3363 [00:00<00:00, 9773.17 examples/s]
# Adding EOS to train dataset: 100%|█████████████████████████████████████████████████████████████████████████████| 3363/3363 [00:00<00:00, 15923.53 examples/s]
# Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 3363/3363 [00:05<00:00, 655.79 examples/s]
# Truncating train dataset: 100%|████████████████████████████████████████████████████████████████████████████████| 3363/3363 [00:00<00:00, 54595.03 examples/s]
# No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
#   0%|                                                                                                                               | 0/2523 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
# {'loss': 0.5775, 'grad_norm': 0.0, 'learning_rate': 7.1146245059288545e-06, 'num_tokens': 35334.0, 'mean_token_accuracy': 0.8757083550095558, 'epoch': 0.01} 
# {'loss': 0.6152, 'grad_norm': 0.0, 'learning_rate': 1.5019762845849802e-05, 'num_tokens': 73349.0, 'mean_token_accuracy': 0.8661734119057656, 'epoch': 0.02} 
# {'loss': 0.5875, 'grad_norm': 0.0, 'learning_rate': 2.2924901185770752e-05, 'num_tokens': 110533.0, 'mean_token_accuracy': 0.8713524967432023, 'epoch': 0.04}
# {'loss': 0.6201, 'grad_norm': 0.0, 'learning_rate': 3.08300395256917e-05, 'num_tokens': 147487.0, 'mean_token_accuracy': 0.8638168767094612, 'epoch': 0.05}  
# {'loss': 0.5731, 'grad_norm': 0.0, 'learning_rate': 3.873517786561265e-05, 'num_tokens': 184722.0, 'mean_token_accuracy': 0.8733947724103928, 'epoch': 0.06}  
# {'loss': 0.6241, 'grad_norm': 0.0, 'learning_rate': 4.66403162055336e-05, 'num_tokens': 220222.0, 'mean_token_accuracy': 0.8636075586080552, 'epoch': 0.07}   
# {'loss': 0.5923, 'grad_norm': 0.0, 'learning_rate': 5.4545454545454546e-05, 'num_tokens': 255060.0, 'mean_token_accuracy': 0.8731164753437042, 'epoch': 0.08} 
# {'loss': 0.5757, 'grad_norm': 0.0, 'learning_rate': 6.245059288537549e-05, 'num_tokens': 292449.0, 'mean_token_accuracy': 0.8744853094220162, 'epoch': 0.1}   
# {'loss': 0.5994, 'grad_norm': 0.0, 'learning_rate': 7.035573122529645e-05, 'num_tokens': 329792.0, 'mean_token_accuracy': 0.8712501659989357, 'epoch': 0.11}  
# {'loss': 0.5821, 'grad_norm': 0.0, 'learning_rate': 7.82608695652174e-05, 'num_tokens': 366028.0, 'mean_token_accuracy': 0.8748950377106667, 'epoch': 0.12}   
# {'loss': 0.599, 'grad_norm': 0.0, 'learning_rate': 8.616600790513835e-05, 'num_tokens': 403720.0, 'mean_token_accuracy': 0.8675450846552849, 'epoch': 0.13}   
# {'loss': 0.5822, 'grad_norm': 0.0, 'learning_rate': 9.407114624505929e-05, 'num_tokens': 441015.0, 'mean_token_accuracy': 0.8709651574492454, 'epoch': 0.14}  
# {'loss': 0.5843, 'grad_norm': 0.0, 'learning_rate': 0.00010197628458498026, 'num_tokens': 476911.0, 'mean_token_accuracy': 0.8748901501297951, 'epoch': 0.15} 
# {'loss': 0.603, 'grad_norm': 0.0, 'learning_rate': 0.0001098814229249012, 'num_tokens': 512954.0, 'mean_token_accuracy': 0.8709266245365143, 'epoch': 0.17}   
# {'loss': 0.5833, 'grad_norm': 0.0, 'learning_rate': 0.00011778656126482215, 'num_tokens': 547726.0, 'mean_token_accuracy': 0.8775181695818901, 'epoch': 0.18} 
# {'loss': 0.6072, 'grad_norm': 0.0, 'learning_rate': 0.0001256916996047431, 'num_tokens': 584446.0, 'mean_token_accuracy': 0.8663908243179321, 'epoch': 0.19}  
# {'loss': 0.5892, 'grad_norm': 0.0, 'learning_rate': 0.00013359683794466405, 'num_tokens': 621309.0, 'mean_token_accuracy': 0.8727191105484963, 'epoch': 0.2}  
# {'loss': 0.5867, 'grad_norm': 0.0, 'learning_rate': 0.00014150197628458498, 'num_tokens': 657726.0, 'mean_token_accuracy': 0.8724945694208145, 'epoch': 0.21} 
# {'loss': 0.6267, 'grad_norm': 0.0, 'learning_rate': 0.00014940711462450593, 'num_tokens': 692812.0, 'mean_token_accuracy': 0.8664441019296646, 'epoch': 0.23} 
# {'loss': 0.5726, 'grad_norm': 0.0, 'learning_rate': 0.00015731225296442689, 'num_tokens': 730726.0, 'mean_token_accuracy': 0.8733742833137512, 'epoch': 0.24} 
# {'loss': 0.5726, 'grad_norm': 0.0, 'learning_rate': 0.00016521739130434784, 'num_tokens': 766986.0, 'mean_token_accuracy': 0.8769846528768539, 'epoch': 0.25} 
# {'loss': 0.5882, 'grad_norm': 0.0, 'learning_rate': 0.0001731225296442688, 'num_tokens': 802924.0, 'mean_token_accuracy': 0.8756162375211716, 'epoch': 0.26}  
# {'loss': 0.6343, 'grad_norm': 0.0, 'learning_rate': 0.00018102766798418972, 'num_tokens': 839406.0, 'mean_token_accuracy': 0.8629887714982033, 'epoch': 0.27} 
# {'loss': 0.6224, 'grad_norm': 0.0, 'learning_rate': 0.00018893280632411067, 'num_tokens': 877080.0, 'mean_token_accuracy': 0.8651782587170601, 'epoch': 0.29} 
# {'loss': 0.6068, 'grad_norm': 0.0, 'learning_rate': 0.00019683794466403162, 'num_tokens': 912362.0, 'mean_token_accuracy': 0.8702954381704331, 'epoch': 0.3}  
# {'loss': 0.5867, 'grad_norm': 0.0, 'learning_rate': 0.00019999655239072328, 'num_tokens': 948188.0, 'mean_token_accuracy': 0.8745792910456658, 'epoch': 0.31} 
# {'loss': 0.5923, 'grad_norm': 0.0, 'learning_rate': 0.00019997548452823914, 'num_tokens': 985748.0, 'mean_token_accuracy': 0.8717014193534851, 'epoch': 0.32} 
# {'loss': 0.6045, 'grad_norm': 0.0, 'learning_rate': 0.00019993526817203407, 'num_tokens': 1024636.0, 'mean_token_accuracy': 0.8600430443882943, 'epoch': 0.33}
# {'loss': 0.5794, 'grad_norm': 0.0, 'learning_rate': 0.0001998759110248229, 'num_tokens': 1061654.0, 'mean_token_accuracy': 0.8709366247057915, 'epoch': 0.34} 
# {'loss': 0.619, 'grad_norm': 0.0, 'learning_rate': 0.00019979742445539231, 'num_tokens': 1095285.0, 'mean_token_accuracy': 0.8692828357219696, 'epoch': 0.36} 
# {'loss': 0.593, 'grad_norm': 0.0, 'learning_rate': 0.00019969982349642344, 'num_tokens': 1131861.0, 'mean_token_accuracy': 0.8683516383171082, 'epoch': 0.37} 
# {'loss': 0.5908, 'grad_norm': 0.0, 'learning_rate': 0.0001995831268416127, 'num_tokens': 1168452.0, 'mean_token_accuracy': 0.8729576379060745, 'epoch': 0.38} 
# {'loss': 0.5736, 'grad_norm': 0.0, 'learning_rate': 0.00019944735684209117, 'num_tokens': 1204879.0, 'mean_token_accuracy': 0.8774775430560112, 'epoch': 0.39}
# {'loss': 0.5866, 'grad_norm': 0.0, 'learning_rate': 0.00019929253950214375, 'num_tokens': 1241986.0, 'mean_token_accuracy': 0.870885843038559, 'epoch': 0.4}  
# {'loss': 0.5923, 'grad_norm': 0.0, 'learning_rate': 0.00019911870447422846, 'num_tokens': 1278583.0, 'mean_token_accuracy': 0.8725741535425187, 'epoch': 0.42}
# {'loss': 0.5561, 'grad_norm': 0.0, 'learning_rate': 0.00019892588505329717, 'num_tokens': 1316593.0, 'mean_token_accuracy': 0.8751594156026841, 'epoch': 0.43}
# {'loss': 0.6045, 'grad_norm': 0.0, 'learning_rate': 0.00019871411817041842, 'num_tokens': 1352935.0, 'mean_token_accuracy': 0.8686731189489365, 'epoch': 0.44}
# {'loss': 0.5909, 'grad_norm': 0.0, 'learning_rate': 0.00019848344438570394, 'num_tokens': 1388071.0, 'mean_token_accuracy': 0.8722481951117516, 'epoch': 0.45}
# {'loss': 0.5935, 'grad_norm': 0.0, 'learning_rate': 0.00019823390788054025, 'num_tokens': 1424969.0, 'mean_token_accuracy': 0.8710988134145736, 'epoch': 0.46}
# {'loss': 0.5924, 'grad_norm': 0.0, 'learning_rate': 0.00019796555644912628, 'num_tokens': 1461050.0, 'mean_token_accuracy': 0.8729088231921196, 'epoch': 0.48}
# {'loss': 0.5773, 'grad_norm': 0.0, 'learning_rate': 0.00019767844148931948, 'num_tokens': 1496612.0, 'mean_token_accuracy': 0.8752109378576278, 'epoch': 0.49}
# {'loss': 0.6089, 'grad_norm': 0.0, 'learning_rate': 0.00019737261799279137, 'num_tokens': 1533621.0, 'mean_token_accuracy': 0.866161373257637, 'epoch': 0.5}  
# {'loss': 0.5747, 'grad_norm': 0.0, 'learning_rate': 0.0001970481445344949, 'num_tokens': 1568750.0, 'mean_token_accuracy': 0.8781893730163575, 'epoch': 0.51} 
# {'loss': 0.5793, 'grad_norm': 0.0, 'learning_rate': 0.00019670508326144552, 'num_tokens': 1604956.0, 'mean_token_accuracy': 0.8723375529050827, 'epoch': 0.52}
# {'loss': 0.5943, 'grad_norm': 0.0, 'learning_rate': 0.00019634349988081792, 'num_tokens': 1642310.0, 'mean_token_accuracy': 0.8710424676537514, 'epoch': 0.54}
# {'loss': 0.6102, 'grad_norm': 0.0, 'learning_rate': 0.00019596346364736123, 'num_tokens': 1679343.0, 'mean_token_accuracy': 0.8661708652973175, 'epoch': 0.55}
# {'loss': 0.6132, 'grad_norm': 0.0, 'learning_rate': 0.00019556504735013431, 'num_tokens': 1714478.0, 'mean_token_accuracy': 0.8705276265740395, 'epoch': 0.56}
# {'loss': 0.5636, 'grad_norm': 0.0, 'learning_rate': 0.0001951483272985644, 'num_tokens': 1749841.0, 'mean_token_accuracy': 0.879919646680355, 'epoch': 0.57}  
# {'loss': 0.561, 'grad_norm': 0.0, 'learning_rate': 0.00019471338330783152, 'num_tokens': 1785684.0, 'mean_token_accuracy': 0.8779202222824096, 'epoch': 0.58} 
# {'loss': 0.5634, 'grad_norm': 0.0, 'learning_rate': 0.00019426029868358118, 'num_tokens': 1822972.0, 'mean_token_accuracy': 0.8800695836544037, 'epoch': 0.59}
# {'loss': 0.5653, 'grad_norm': 0.0, 'learning_rate': 0.00019378916020596878, 'num_tokens': 1859245.0, 'mean_token_accuracy': 0.878256069123745, 'epoch': 0.61} 
# {'loss': 0.5799, 'grad_norm': 0.0, 'learning_rate': 0.0001933000581130384, 'num_tokens': 1895501.0, 'mean_token_accuracy': 0.873122563958168, 'epoch': 0.62}  
# {'loss': 0.6276, 'grad_norm': 0.0, 'learning_rate': 0.00019279308608343934, 'num_tokens': 1931041.0, 'mean_token_accuracy': 0.8616216123104096, 'epoch': 0.63}
# {'loss': 0.573, 'grad_norm': 0.0, 'learning_rate': 0.00019226834121848372, 'num_tokens': 1968689.0, 'mean_token_accuracy': 0.874072627723217, 'epoch': 0.64}  
# {'loss': 0.5552, 'grad_norm': 0.0, 'learning_rate': 0.00019172592402354842, 'num_tokens': 2006400.0, 'mean_token_accuracy': 0.87906863540411, 'epoch': 0.65}  
# {'loss': 0.593, 'grad_norm': 0.0, 'learning_rate': 0.0001911659383888251, 'num_tokens': 2042659.0, 'mean_token_accuracy': 0.8725034058094024, 'epoch': 0.67}  
# {'loss': 0.5936, 'grad_norm': 0.0, 'learning_rate': 0.00019058849156942197, 'num_tokens': 2079034.0, 'mean_token_accuracy': 0.8710827738046646, 'epoch': 0.68}
# {'loss': 0.6062, 'grad_norm': 0.0, 'learning_rate': 0.000189993694164821, 'num_tokens': 2114592.0, 'mean_token_accuracy': 0.8669642746448517, 'epoch': 0.69}  
# {'loss': 0.6052, 'grad_norm': 0.0, 'learning_rate': 0.00018938166009769452, 'num_tokens': 2150983.0, 'mean_token_accuracy': 0.8674422889947891, 'epoch': 0.7} 
# {'loss': 0.5807, 'grad_norm': 0.0, 'learning_rate': 0.00018875250659208546, 'num_tokens': 2187679.0, 'mean_token_accuracy': 0.8755146920681, 'epoch': 0.71}   
# {'loss': 0.5808, 'grad_norm': 0.0, 'learning_rate': 0.0001881063541509552, 'num_tokens': 2222811.0, 'mean_token_accuracy': 0.8723723575472832, 'epoch': 0.73} 
# {'loss': 0.5487, 'grad_norm': 0.0, 'learning_rate': 0.00018744332653310348, 'num_tokens': 2259755.0, 'mean_token_accuracy': 0.881856894493103, 'epoch': 0.74} 
# {'loss': 0.5873, 'grad_norm': 0.0, 'learning_rate': 0.00018676355072946442, 'num_tokens': 2297417.0, 'mean_token_accuracy': 0.8709309220314025, 'epoch': 0.75}
# {'loss': 0.5932, 'grad_norm': 0.0, 'learning_rate': 0.00018606715693878396, 'num_tokens': 2332931.0, 'mean_token_accuracy': 0.8708799123764038, 'epoch': 0.76}
# {'loss': 0.5984, 'grad_norm': 0.0, 'learning_rate': 0.00018535427854268253, 'num_tokens': 2368647.0, 'mean_token_accuracy': 0.8688210308551788, 'epoch': 0.77}
# {'loss': 0.6021, 'grad_norm': 0.0, 'learning_rate': 0.00018462505208010819, 'num_tokens': 2405346.0, 'mean_token_accuracy': 0.8671869784593582, 'epoch': 0.79}
# {'loss': 0.5664, 'grad_norm': 0.0, 'learning_rate': 0.00018387961722118512, 'num_tokens': 2442040.0, 'mean_token_accuracy': 0.8794109940528869, 'epoch': 0.8} 
# {'loss': 0.5884, 'grad_norm': 0.0, 'learning_rate': 0.00018311811674046234, 'num_tokens': 2478217.0, 'mean_token_accuracy': 0.8717756375670433, 'epoch': 0.81}
# {'loss': 0.6009, 'grad_norm': 0.0, 'learning_rate': 0.00018234069648956783, 'num_tokens': 2515713.0, 'mean_token_accuracy': 0.8667199552059174, 'epoch': 0.82}
# {'loss': 0.603, 'grad_norm': 0.0, 'learning_rate': 0.00018154750536927323, 'num_tokens': 2552241.0, 'mean_token_accuracy': 0.8672344341874123, 'epoch': 0.83} 
# {'loss': 0.6068, 'grad_norm': 0.0, 'learning_rate': 0.0001807386953009746, 'num_tokens': 2590467.0, 'mean_token_accuracy': 0.8655141219496727, 'epoch': 0.84} 
# {'loss': 0.5829, 'grad_norm': 0.0, 'learning_rate': 0.00017991442119759477, 'num_tokens': 2626854.0, 'mean_token_accuracy': 0.8739048078656196, 'epoch': 0.86}
# {'loss': 0.576, 'grad_norm': 0.0, 'learning_rate': 0.00017907484093391242, 'num_tokens': 2663705.0, 'mean_token_accuracy': 0.8774152874946595, 'epoch': 0.87} 
# {'loss': 0.555, 'grad_norm': 0.0, 'learning_rate': 0.00017822011531632405, 'num_tokens': 2700505.0, 'mean_token_accuracy': 0.8812798887491227, 'epoch': 0.88} 
# {'loss': 0.585, 'grad_norm': 0.0, 'learning_rate': 0.00017735040805204447, 'num_tokens': 2737841.0, 'mean_token_accuracy': 0.8706025511026383, 'epoch': 0.89} 
# {'loss': 0.5742, 'grad_norm': 0.0, 'learning_rate': 0.0001764658857177516, 'num_tokens': 2774579.0, 'mean_token_accuracy': 0.8787211641669274, 'epoch': 0.9}  
# {'loss': 0.5902, 'grad_norm': 0.0, 'learning_rate': 0.00017556671772768186, 'num_tokens': 2810309.0, 'mean_token_accuracy': 0.8735142707824707, 'epoch': 0.92}
# {'loss': 0.6048, 'grad_norm': 0.0, 'learning_rate': 0.0001746530763011817, 'num_tokens': 2847505.0, 'mean_token_accuracy': 0.8669482856988907, 'epoch': 0.93} 
# {'loss': 0.5671, 'grad_norm': 0.0, 'learning_rate': 0.00017372513642972226, 'num_tokens': 2885462.0, 'mean_token_accuracy': 0.875202564895153, 'epoch': 0.94} 
# {'loss': 0.5979, 'grad_norm': 0.0, 'learning_rate': 0.000172783075843383, 'num_tokens': 2920228.0, 'mean_token_accuracy': 0.8734383180737495, 'epoch': 0.95}  
# {'loss': 0.648, 'grad_norm': 0.0, 'learning_rate': 0.0001718270749768105, 'num_tokens': 2958413.0, 'mean_token_accuracy': 0.8577236667275429, 'epoch': 0.96}  
# {'loss': 0.5927, 'grad_norm': 0.0, 'learning_rate': 0.00017085731693465965, 'num_tokens': 2995181.0, 'mean_token_accuracy': 0.8733200401067733, 'epoch': 0.98}
# {'loss': 0.5684, 'grad_norm': 0.0, 'learning_rate': 0.0001698739874565232, 'num_tokens': 3031315.0, 'mean_token_accuracy': 0.8778680175542831, 'epoch': 0.99} 
# {'loss': 0.5566, 'grad_norm': 0.0, 'learning_rate': 0.00016887727488135672, 'num_tokens': 3069340.0, 'mean_token_accuracy': 0.8768566861748696, 'epoch': 1.0} 
#  33%|█████████████████████████████████████▋                                                                           | 841/2523 [2:22:57<4:22:37,  9.37s/it]C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\other.py:1221: UserWarning: Unable to fetch remote file due to the following error (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1007)')))"), '(Request ID: d2ddd41a-8358-4694-932e-f9d00c91b1e3)') - silently ignoring the lookup for the file config.json in Qwen/Qwen2.5-Coder-3B-Instruct.
#   warnings.warn(
# C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\save_and_load.py:238: UserWarning: Could not find a config file in Qwen/Qwen2.5-Coder-3B-Instruct - will assume that the vocabulary was not modified.
#   warnings.warn(
# {'loss': 0.5802, 'grad_norm': 0.0, 'learning_rate': 0.00016786737011140565, 'num_tokens': 3103586.0, 'mean_token_accuracy': 0.8797828280008756, 'epoch': 1.01}
# {'loss': 0.566, 'grad_norm': 0.0, 'learning_rate': 0.0001668444665756415, 'num_tokens': 3141800.0, 'mean_token_accuracy': 0.8752215370535851, 'epoch': 1.02}  
# {'loss': 0.5761, 'grad_norm': 0.0, 'learning_rate': 0.0001658087601927139, 'num_tokens': 3178207.0, 'mean_token_accuracy': 0.8718273043632507, 'epoch': 1.03} 
# {'loss': 0.5652, 'grad_norm': 0.0, 'learning_rate': 0.0001647604493334262, 'num_tokens': 3216326.0, 'mean_token_accuracy': 0.875446155667305, 'epoch': 1.05}  
# {'loss': 0.5985, 'grad_norm': 0.0, 'learning_rate': 0.00016369973478274073, 'num_tokens': 3252213.0, 'mean_token_accuracy': 0.8702464699745178, 'epoch': 1.06}
# {'loss': 0.6059, 'grad_norm': 0.0, 'learning_rate': 0.0001626268197013225, 'num_tokens': 3287980.0, 'mean_token_accuracy': 0.8694165751338006, 'epoch': 1.07} 
# {'loss': 0.5738, 'grad_norm': 0.0, 'learning_rate': 0.00016154190958662717, 'num_tokens': 3325071.0, 'mean_token_accuracy': 0.8749718502163887, 'epoch': 1.08}
# {'loss': 0.5967, 'grad_norm': 0.0, 'learning_rate': 0.00016044521223354178, 'num_tokens': 3362221.0, 'mean_token_accuracy': 0.8679421097040176, 'epoch': 1.09}
# {'loss': 0.594, 'grad_norm': 0.0, 'learning_rate': 0.00015933693769458555, 'num_tokens': 3396755.0, 'mean_token_accuracy': 0.8741679102182388, 'epoch': 1.11} 
# {'loss': 0.6188, 'grad_norm': 0.0, 'learning_rate': 0.0001582172982396779, 'num_tokens': 3433223.0, 'mean_token_accuracy': 0.8651689141988754, 'epoch': 1.12} 
# {'loss': 0.6031, 'grad_norm': 0.0, 'learning_rate': 0.00015708650831548223, 'num_tokens': 3468774.0, 'mean_token_accuracy': 0.8715772747993469, 'epoch': 1.13}
# {'loss': 0.6, 'grad_norm': 0.0, 'learning_rate': 0.00015594478450433225, 'num_tokens': 3505062.0, 'mean_token_accuracy': 0.8725473061203957, 'epoch': 1.14}   
# {'loss': 0.6046, 'grad_norm': 0.0, 'learning_rate': 0.00015479234548274992, 'num_tokens': 3541458.0, 'mean_token_accuracy': 0.8691560119390488, 'epoch': 1.15}
# {'loss': 0.5725, 'grad_norm': 0.0, 'learning_rate': 0.00015362941197956165, 'num_tokens': 3578719.0, 'mean_token_accuracy': 0.8760386720299721, 'epoch': 1.17}
# {'loss': 0.5892, 'grad_norm': 0.0, 'learning_rate': 0.00015245620673362163, 'num_tokens': 3615414.0, 'mean_token_accuracy': 0.8734348371624947, 'epoch': 1.18}
# {'loss': 0.5821, 'grad_norm': 0.0, 'learning_rate': 0.00015127295445115064, 'num_tokens': 3653086.0, 'mean_token_accuracy': 0.8724568009376525, 'epoch': 1.19}
# {'loss': 0.5985, 'grad_norm': 0.0, 'learning_rate': 0.00015007988176269716, 'num_tokens': 3690175.0, 'mean_token_accuracy': 0.8684394046664238, 'epoch': 1.2} 
# {'loss': 0.5825, 'grad_norm': 0.0, 'learning_rate': 0.00014887721717973072, 'num_tokens': 3726902.0, 'mean_token_accuracy': 0.873192822933197, 'epoch': 1.21} 
# {'loss': 0.6052, 'grad_norm': 0.0, 'learning_rate': 0.00014766519105087453, 'num_tokens': 3762639.0, 'mean_token_accuracy': 0.8681438460946083, 'epoch': 1.22}
# {'loss': 0.5682, 'grad_norm': 0.0, 'learning_rate': 0.00014644403551778638, 'num_tokens': 3801909.0, 'mean_token_accuracy': 0.8718546241521835, 'epoch': 1.24}
# {'loss': 0.5995, 'grad_norm': 0.0, 'learning_rate': 0.00014521398447069615, 'num_tokens': 3836389.0, 'mean_token_accuracy': 0.8737059533596039, 'epoch': 1.25}
# {'loss': 0.5733, 'grad_norm': 0.0, 'learning_rate': 0.00014397527350360831, 'num_tokens': 3872801.0, 'mean_token_accuracy': 0.8765065252780915, 'epoch': 1.26}
# {'loss': 0.5896, 'grad_norm': 0.0, 'learning_rate': 0.00014272813986917826, 'num_tokens': 3908878.0, 'mean_token_accuracy': 0.8702822342514992, 'epoch': 1.27}
# {'loss': 0.6313, 'grad_norm': 0.0, 'learning_rate': 0.00014147282243327073, 'num_tokens': 3943426.0, 'mean_token_accuracy': 0.8641292154788971, 'epoch': 1.28}
# {'loss': 0.5837, 'grad_norm': 0.0, 'learning_rate': 0.0001402095616292095, 'num_tokens': 3980805.0, 'mean_token_accuracy': 0.8719038560986518, 'epoch': 1.3}  
# {'loss': 0.6342, 'grad_norm': 0.0, 'learning_rate': 0.00013893859941172668, 'num_tokens': 4017003.0, 'mean_token_accuracy': 0.8642883494496345, 'epoch': 1.31}
# {'loss': 0.5692, 'grad_norm': 0.0, 'learning_rate': 0.0001376601792106206, 'num_tokens': 4053822.0, 'mean_token_accuracy': 0.878726176917553, 'epoch': 1.32}  
# {'loss': 0.5795, 'grad_norm': 0.0, 'learning_rate': 0.0001363745458841314, 'num_tokens': 4091995.0, 'mean_token_accuracy': 0.8707404568791389, 'epoch': 1.33} 
# {'loss': 0.5777, 'grad_norm': 0.0, 'learning_rate': 0.00013508194567204266, 'num_tokens': 4128577.0, 'mean_token_accuracy': 0.8761247977614403, 'epoch': 1.34}
# {'loss': 0.5749, 'grad_norm': 0.0, 'learning_rate': 0.00013378262614851887, 'num_tokens': 4165512.0, 'mean_token_accuracy': 0.8754744395613671, 'epoch': 1.36}
# {'loss': 0.5986, 'grad_norm': 0.0, 'learning_rate': 0.0001324768361746868, 'num_tokens': 4202000.0, 'mean_token_accuracy': 0.8686405539512634, 'epoch': 1.37} 
# {'loss': 0.607, 'grad_norm': 0.0, 'learning_rate': 0.00013116482585097102, 'num_tokens': 4239227.0, 'mean_token_accuracy': 0.8678390607237816, 'epoch': 1.38} 
# {'loss': 0.6132, 'grad_norm': 0.0, 'learning_rate': 0.0001298468464691912, 'num_tokens': 4274730.0, 'mean_token_accuracy': 0.8673344790935517, 'epoch': 1.39} 
# {'loss': 0.5566, 'grad_norm': 0.0, 'learning_rate': 0.0001285231504644323, 'num_tokens': 4310911.0, 'mean_token_accuracy': 0.8806039854884148, 'epoch': 1.4}  
# {'loss': 0.6102, 'grad_norm': 0.0, 'learning_rate': 0.0001271939913666947, 'num_tokens': 4346226.0, 'mean_token_accuracy': 0.8699426710605621, 'epoch': 1.42} 
# {'loss': 0.5887, 'grad_norm': 0.0, 'learning_rate': 0.00012585962375233532, 'num_tokens': 4382203.0, 'mean_token_accuracy': 0.876276932656765, 'epoch': 1.43} 
# {'loss': 0.6317, 'grad_norm': 0.0, 'learning_rate': 0.00012452030319530825, 'num_tokens': 4417978.0, 'mean_token_accuracy': 0.861655643582344, 'epoch': 1.44} 
# {'loss': 0.5976, 'grad_norm': 0.0, 'learning_rate': 0.00012317628621821388, 'num_tokens': 4455611.0, 'mean_token_accuracy': 0.8681260034441948, 'epoch': 1.45}
# {'loss': 0.6123, 'grad_norm': 0.0, 'learning_rate': 0.00012182783024316705, 'num_tokens': 4491400.0, 'mean_token_accuracy': 0.8658653363585472, 'epoch': 1.46}
# {'loss': 0.5802, 'grad_norm': 0.0, 'learning_rate': 0.00012047519354249207, 'num_tokens': 4526201.0, 'mean_token_accuracy': 0.8754538729786873, 'epoch': 1.47}
# {'loss': 0.5631, 'grad_norm': 0.0, 'learning_rate': 0.00011911863518925561, 'num_tokens': 4562719.0, 'mean_token_accuracy': 0.8785970658063889, 'epoch': 1.49}
# {'loss': 0.563, 'grad_norm': 0.0, 'learning_rate': 0.00011775841500764593, 'num_tokens': 4599300.0, 'mean_token_accuracy': 0.8768683061003685, 'epoch': 1.5}  
# {'loss': 0.5916, 'grad_norm': 0.0, 'learning_rate': 0.0001163947935232081, 'num_tokens': 4635048.0, 'mean_token_accuracy': 0.8727613627910614, 'epoch': 1.51} 
# {'loss': 0.6016, 'grad_norm': 0.0, 'learning_rate': 0.0001150280319129453, 'num_tokens': 4671599.0, 'mean_token_accuracy': 0.8695131048560143, 'epoch': 1.52} 
# {'loss': 0.6176, 'grad_norm': 0.0, 'learning_rate': 0.00011365839195529484, 'num_tokens': 4706803.0, 'mean_token_accuracy': 0.8671004191040993, 'epoch': 1.53}
# {'loss': 0.5643, 'grad_norm': 0.0, 'learning_rate': 0.00011228613597998934, 'num_tokens': 4742774.0, 'mean_token_accuracy': 0.8776543453335762, 'epoch': 1.55}
# {'loss': 0.5933, 'grad_norm': 0.0, 'learning_rate': 0.00011091152681781234, 'num_tokens': 4780446.0, 'mean_token_accuracy': 0.8686742141842843, 'epoch': 1.56}
# {'loss': 0.5982, 'grad_norm': 0.0, 'learning_rate': 0.00010953482775025759, 'num_tokens': 4816202.0, 'mean_token_accuracy': 0.8687623739242554, 'epoch': 1.57}
# {'loss': 0.5881, 'grad_norm': 0.0, 'learning_rate': 0.00010815630245910244, 'num_tokens': 4852263.0, 'mean_token_accuracy': 0.8753452256321907, 'epoch': 1.58}
# {'loss': 0.6089, 'grad_norm': 0.0, 'learning_rate': 0.00010677621497590432, 'num_tokens': 4889934.0, 'mean_token_accuracy': 0.8664796561002731, 'epoch': 1.59}
# {'loss': 0.5794, 'grad_norm': 0.0, 'learning_rate': 0.00010539482963143019, 'num_tokens': 4926166.0, 'mean_token_accuracy': 0.8749096229672432, 'epoch': 1.61}
# {'loss': 0.6045, 'grad_norm': 0.0, 'learning_rate': 0.0001040124110050289, 'num_tokens': 4963540.0, 'mean_token_accuracy': 0.8681509822607041, 'epoch': 1.62} 
# {'loss': 0.5896, 'grad_norm': 0.0, 'learning_rate': 0.00010262922387395573, 'num_tokens': 5000124.0, 'mean_token_accuracy': 0.8684981778264046, 'epoch': 1.63}
# {'loss': 0.576, 'grad_norm': 0.0, 'learning_rate': 0.00010124553316265905, 'num_tokens': 5036277.0, 'mean_token_accuracy': 0.8757539182901383, 'epoch': 1.64} 
# {'loss': 0.6267, 'grad_norm': 0.0, 'learning_rate': 9.986160389203898e-05, 'num_tokens': 5072212.0, 'mean_token_accuracy': 0.8612911999225616, 'epoch': 1.65} 
# {'loss': 0.5889, 'grad_norm': 0.0, 'learning_rate': 9.847770112868735e-05, 'num_tokens': 5109200.0, 'mean_token_accuracy': 0.8700929403305053, 'epoch': 1.66} 
# {'loss': 0.6297, 'grad_norm': 0.0, 'learning_rate': 9.709408993411898e-05, 'num_tokens': 5145390.0, 'mean_token_accuracy': 0.8644052773714066, 'epoch': 1.68} 
# {'loss': 0.622, 'grad_norm': 0.0, 'learning_rate': 9.571103531400394e-05, 'num_tokens': 5181981.0, 'mean_token_accuracy': 0.8629093527793884, 'epoch': 1.69}  
# {'loss': 0.5854, 'grad_norm': 0.0, 'learning_rate': 9.432880216741063e-05, 'num_tokens': 5219166.0, 'mean_token_accuracy': 0.8747522935271264, 'epoch': 1.7}  
# {'loss': 0.58, 'grad_norm': 0.0, 'learning_rate': 9.294765523606894e-05, 'num_tokens': 5254425.0, 'mean_token_accuracy': 0.8787077039480209, 'epoch': 1.71}   
# {'loss': 0.6007, 'grad_norm': 0.0, 'learning_rate': 9.156785905366406e-05, 'num_tokens': 5290825.0, 'mean_token_accuracy': 0.8710288614034652, 'epoch': 1.72} 
# {'loss': 0.5592, 'grad_norm': 0.0, 'learning_rate': 9.018967789516956e-05, 'num_tokens': 5327405.0, 'mean_token_accuracy': 0.8790363654494285, 'epoch': 1.74} 
# {'loss': 0.599, 'grad_norm': 0.0, 'learning_rate': 8.881337572623045e-05, 'num_tokens': 5362778.0, 'mean_token_accuracy': 0.871659904718399, 'epoch': 1.75}   
# {'loss': 0.6121, 'grad_norm': 0.0, 'learning_rate': 8.743921615260532e-05, 'num_tokens': 5399270.0, 'mean_token_accuracy': 0.8653246372938156, 'epoch': 1.76} 
# {'loss': 0.5898, 'grad_norm': 0.0, 'learning_rate': 8.606746236967749e-05, 'num_tokens': 5436288.0, 'mean_token_accuracy': 0.8708102688193321, 'epoch': 1.77} 
# {'loss': 0.5571, 'grad_norm': 0.0, 'learning_rate': 8.46983771120445e-05, 'num_tokens': 5472528.0, 'mean_token_accuracy': 0.8813493087887764, 'epoch': 1.78}  
# {'loss': 0.6051, 'grad_norm': 0.0, 'learning_rate': 8.333222260319627e-05, 'num_tokens': 5508426.0, 'mean_token_accuracy': 0.8700441777706146, 'epoch': 1.8}  
# {'loss': 0.6204, 'grad_norm': 0.0, 'learning_rate': 8.196926050529091e-05, 'num_tokens': 5546916.0, 'mean_token_accuracy': 0.8603341430425644, 'epoch': 1.81} 
# {'loss': 0.6071, 'grad_norm': 0.0, 'learning_rate': 8.060975186903799e-05, 'num_tokens': 5584584.0, 'mean_token_accuracy': 0.8636341452598572, 'epoch': 1.82} 
# {'loss': 0.5466, 'grad_norm': 0.0, 'learning_rate': 7.925395708369892e-05, 'num_tokens': 5621315.0, 'mean_token_accuracy': 0.8832762688398361, 'epoch': 1.83} 
# {'loss': 0.5712, 'grad_norm': 0.0, 'learning_rate': 7.790213582721437e-05, 'num_tokens': 5657965.0, 'mean_token_accuracy': 0.8757675066590309, 'epoch': 1.84} 
# {'loss': 0.56, 'grad_norm': 0.0, 'learning_rate': 7.655454701646748e-05, 'num_tokens': 5694625.0, 'mean_token_accuracy': 0.8792400434613228, 'epoch': 1.86}   
# {'loss': 0.5776, 'grad_norm': 0.0, 'learning_rate': 7.521144875769293e-05, 'num_tokens': 5731063.0, 'mean_token_accuracy': 0.8731464818120003, 'epoch': 1.87} 
# {'loss': 0.5851, 'grad_norm': 0.0, 'learning_rate': 7.387309829704165e-05, 'num_tokens': 5767754.0, 'mean_token_accuracy': 0.8738893702626228, 'epoch': 1.88} 
# {'loss': 0.6124, 'grad_norm': 0.0, 'learning_rate': 7.253975197130976e-05, 'num_tokens': 5803818.0, 'mean_token_accuracy': 0.868033504486084, 'epoch': 1.89}  
# {'loss': 0.5538, 'grad_norm': 0.0, 'learning_rate': 7.121166515884182e-05, 'num_tokens': 5839586.0, 'mean_token_accuracy': 0.881498996913433, 'epoch': 1.9}   
# {'loss': 0.5641, 'grad_norm': 0.0, 'learning_rate': 6.988909223061803e-05, 'num_tokens': 5874788.0, 'mean_token_accuracy': 0.8825106620788574, 'epoch': 1.91} 
# {'loss': 0.601, 'grad_norm': 0.0, 'learning_rate': 6.857228650153392e-05, 'num_tokens': 5912991.0, 'mean_token_accuracy': 0.8643366396427155, 'epoch': 1.93}  
# {'loss': 0.5944, 'grad_norm': 0.0, 'learning_rate': 6.726150018188223e-05, 'num_tokens': 5951125.0, 'mean_token_accuracy': 0.8686752751469612, 'epoch': 1.94} 
# {'loss': 0.5559, 'grad_norm': 0.0, 'learning_rate': 6.595698432904708e-05, 'num_tokens': 5988500.0, 'mean_token_accuracy': 0.8797275394201278, 'epoch': 1.95} 
# {'loss': 0.573, 'grad_norm': 0.0, 'learning_rate': 6.465898879941808e-05, 'num_tokens': 6026398.0, 'mean_token_accuracy': 0.8731493785977363, 'epoch': 1.96}  
# {'loss': 0.603, 'grad_norm': 0.0, 'learning_rate': 6.336776220053509e-05, 'num_tokens': 6063221.0, 'mean_token_accuracy': 0.8688651517033577, 'epoch': 1.97}  
# {'loss': 0.584, 'grad_norm': 0.0, 'learning_rate': 6.20835518434717e-05, 'num_tokens': 6099301.0, 'mean_token_accuracy': 0.8734522372484207, 'epoch': 1.99}   
# {'loss': 0.5845, 'grad_norm': 0.0, 'learning_rate': 6.080660369546758e-05, 'num_tokens': 6138124.0, 'mean_token_accuracy': 0.87098268866539, 'epoch': 2.0}    
#  67%|██████████████████████████████████████████████████████████████████████████▋                                     | 1682/2523 [4:39:49<2:09:47,  9.26s/it]C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\other.py:1221: UserWarning: Unable to fetch remote file due to the following error (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1007)')))"), '(Request ID: c89072e9-3ab0-46f5-bce6-2f7a26418343)') - silently ignoring the lookup for the file config.json in Qwen/Qwen2.5-Coder-3B-Instruct.
#   warnings.warn(
# C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\save_and_load.py:238: UserWarning: Could not find a config file in Qwen/Qwen2.5-Coder-3B-Instruct - will assume that the vocabulary was not modified.
#   warnings.warn(
# {'loss': 0.5866, 'grad_norm': 0.0, 'learning_rate': 5.9537162332817696e-05, 'num_tokens': 6173923.0, 'mean_token_accuracy': 0.8693164678720328, 'epoch': 2.01}
# {'loss': 0.6221, 'grad_norm': 0.0, 'learning_rate': 5.8275470894028205e-05, 'num_tokens': 6208393.0, 'mean_token_accuracy': 0.8697834447026253, 'epoch': 2.02}
# {'loss': 0.6051, 'grad_norm': 0.0, 'learning_rate': 5.7021771033247694e-05, 'num_tokens': 6244975.0, 'mean_token_accuracy': 0.870494844019413, 'epoch': 2.03} 
# {'loss': 0.5637, 'grad_norm': 0.0, 'learning_rate': 5.5776302873982746e-05, 'num_tokens': 6282696.0, 'mean_token_accuracy': 0.8773292660713196, 'epoch': 2.05}
# {'loss': 0.5737, 'grad_norm': 0.0, 'learning_rate': 5.453930496310645e-05, 'num_tokens': 6319160.0, 'mean_token_accuracy': 0.8794679313898086, 'epoch': 2.06} 
# {'loss': 0.5724, 'grad_norm': 0.0, 'learning_rate': 5.3311014225169306e-05, 'num_tokens': 6355627.0, 'mean_token_accuracy': 0.8766194865107536, 'epoch': 2.07}
# {'loss': 0.6025, 'grad_norm': 0.0, 'learning_rate': 5.209166591702051e-05, 'num_tokens': 6393560.0, 'mean_token_accuracy': 0.8657386183738709, 'epoch': 2.08} 
# {'loss': 0.6086, 'grad_norm': 0.0, 'learning_rate': 5.088149358274862e-05, 'num_tokens': 6431137.0, 'mean_token_accuracy': 0.8662982612848282, 'epoch': 2.09} 
# {'loss': 0.572, 'grad_norm': 0.0, 'learning_rate': 4.968072900895072e-05, 'num_tokens': 6468872.0, 'mean_token_accuracy': 0.8747387930750847, 'epoch': 2.1}   
# {'loss': 0.5896, 'grad_norm': 0.0, 'learning_rate': 4.848960218033765e-05, 'num_tokens': 6507819.0, 'mean_token_accuracy': 0.8674436822533608, 'epoch': 2.12} 
# {'loss': 0.5732, 'grad_norm': 0.0, 'learning_rate': 4.730834123568473e-05, 'num_tokens': 6543762.0, 'mean_token_accuracy': 0.8768094584345818, 'epoch': 2.13} 
# {'loss': 0.5875, 'grad_norm': 0.0, 'learning_rate': 4.6137172424135756e-05, 'num_tokens': 6580346.0, 'mean_token_accuracy': 0.8720676898956299, 'epoch': 2.14}
# {'loss': 0.5906, 'grad_norm': 0.0, 'learning_rate': 4.497632006186926e-05, 'num_tokens': 6617868.0, 'mean_token_accuracy': 0.8682911023497581, 'epoch': 2.15} 
# {'loss': 0.5864, 'grad_norm': 0.0, 'learning_rate': 4.3826006489134754e-05, 'num_tokens': 6652449.0, 'mean_token_accuracy': 0.8765302032232285, 'epoch': 2.16}
# {'loss': 0.5953, 'grad_norm': 0.0, 'learning_rate': 4.26864520276673e-05, 'num_tokens': 6688951.0, 'mean_token_accuracy': 0.870103220641613, 'epoch': 2.18}   
# {'loss': 0.5883, 'grad_norm': 0.0, 'learning_rate': 4.1557874938489174e-05, 'num_tokens': 6726603.0, 'mean_token_accuracy': 0.8722784593701363, 'epoch': 2.19}
# {'loss': 0.5776, 'grad_norm': 0.0, 'learning_rate': 4.044049138010575e-05, 'num_tokens': 6763192.0, 'mean_token_accuracy': 0.8756429478526115, 'epoch': 2.2}  
# {'loss': 0.5815, 'grad_norm': 0.0, 'learning_rate': 3.933451536710413e-05, 'num_tokens': 6798882.0, 'mean_token_accuracy': 0.8758707284927368, 'epoch': 2.21} 
# {'loss': 0.6004, 'grad_norm': 0.0, 'learning_rate': 3.8240158729162836e-05, 'num_tokens': 6834422.0, 'mean_token_accuracy': 0.8682443529367447, 'epoch': 2.22}
# {'loss': 0.5753, 'grad_norm': 0.0, 'learning_rate': 3.7157631070479426e-05, 'num_tokens': 6870708.0, 'mean_token_accuracy': 0.875755563378334, 'epoch': 2.24} 
# {'loss': 0.5788, 'grad_norm': 0.0, 'learning_rate': 3.608713972962464e-05, 'num_tokens': 6909032.0, 'mean_token_accuracy': 0.8732170209288597, 'epoch': 2.25} 
# {'loss': 0.592, 'grad_norm': 0.0, 'learning_rate': 3.502888973983055e-05, 'num_tokens': 6943935.0, 'mean_token_accuracy': 0.8713882431387902, 'epoch': 2.26}  
# {'loss': 0.5961, 'grad_norm': 0.0, 'learning_rate': 3.398308378972007e-05, 'num_tokens': 6979785.0, 'mean_token_accuracy': 0.870099051296711, 'epoch': 2.27}  
# {'loss': 0.5821, 'grad_norm': 0.0, 'learning_rate': 3.2949922184485695e-05, 'num_tokens': 7018709.0, 'mean_token_accuracy': 0.8697713941335679, 'epoch': 2.28}
# {'loss': 0.5875, 'grad_norm': 0.0, 'learning_rate': 3.1929602807524504e-05, 'num_tokens': 7054897.0, 'mean_token_accuracy': 0.8712772339582443, 'epoch': 2.29}
# {'loss': 0.6001, 'grad_norm': 0.0, 'learning_rate': 3.092232108253741e-05, 'num_tokens': 7091580.0, 'mean_token_accuracy': 0.8678328856825829, 'epoch': 2.31} 
# {'loss': 0.6027, 'grad_norm': 0.0, 'learning_rate': 2.992826993609916e-05, 'num_tokens': 7127024.0, 'mean_token_accuracy': 0.8694511145353317, 'epoch': 2.32} 
# {'loss': 0.6005, 'grad_norm': 0.0, 'learning_rate': 2.894763976070668e-05, 'num_tokens': 7162746.0, 'mean_token_accuracy': 0.8705486819148064, 'epoch': 2.33} 
# {'loss': 0.6081, 'grad_norm': 0.0, 'learning_rate': 2.7980618378312985e-05, 'num_tokens': 7197931.0, 'mean_token_accuracy': 0.8706071496009826, 'epoch': 2.34}
# {'loss': 0.6346, 'grad_norm': 0.0, 'learning_rate': 2.702739100435323e-05, 'num_tokens': 7234074.0, 'mean_token_accuracy': 0.8598601311445236, 'epoch': 2.35} 
# {'loss': 0.5943, 'grad_norm': 0.0, 'learning_rate': 2.608814021226996e-05, 'num_tokens': 7271235.0, 'mean_token_accuracy': 0.871102836728096, 'epoch': 2.37}  
# {'loss': 0.5973, 'grad_norm': 0.0, 'learning_rate': 2.516304589854461e-05, 'num_tokens': 7307153.0, 'mean_token_accuracy': 0.8710170194506646, 'epoch': 2.38} 
# {'loss': 0.5726, 'grad_norm': 0.0, 'learning_rate': 2.4252285248241546e-05, 'num_tokens': 7344667.0, 'mean_token_accuracy': 0.8761409029364586, 'epoch': 2.39}
# {'loss': 0.5802, 'grad_norm': 0.0, 'learning_rate': 2.335603270107136e-05, 'num_tokens': 7379788.0, 'mean_token_accuracy': 0.874775318801403, 'epoch': 2.4}   
# {'loss': 0.6105, 'grad_norm': 0.0, 'learning_rate': 2.2474459917980296e-05, 'num_tokens': 7416132.0, 'mean_token_accuracy': 0.8653220474720001, 'epoch': 2.41}
# {'loss': 0.5929, 'grad_norm': 0.0, 'learning_rate': 2.1607735748271564e-05, 'num_tokens': 7452030.0, 'mean_token_accuracy': 0.8760874137282372, 'epoch': 2.43}
# {'loss': 0.5675, 'grad_norm': 0.0, 'learning_rate': 2.07560261972654e-05, 'num_tokens': 7489003.0, 'mean_token_accuracy': 0.874991549551487, 'epoch': 2.44}   
# {'loss': 0.5989, 'grad_norm': 0.0, 'learning_rate': 1.991949439450361e-05, 'num_tokens': 7525546.0, 'mean_token_accuracy': 0.8727032616734505, 'epoch': 2.45} 
# {'loss': 0.5669, 'grad_norm': 0.0, 'learning_rate': 1.9098300562505266e-05, 'num_tokens': 7563969.0, 'mean_token_accuracy': 0.8750457555055619, 'epoch': 2.46}
# {'loss': 0.5915, 'grad_norm': 0.0, 'learning_rate': 1.829260198607885e-05, 'num_tokens': 7601247.0, 'mean_token_accuracy': 0.868541133403778, 'epoch': 2.47}  
# {'loss': 0.5827, 'grad_norm': 0.0, 'learning_rate': 1.7502552982197186e-05, 'num_tokens': 7636025.0, 'mean_token_accuracy': 0.8769725292921067, 'epoch': 2.49}
# {'loss': 0.5838, 'grad_norm': 0.0, 'learning_rate': 1.672830487044088e-05, 'num_tokens': 7672788.0, 'mean_token_accuracy': 0.8733032405376434, 'epoch': 2.5}  
# {'loss': 0.5783, 'grad_norm': 0.0, 'learning_rate': 1.5970005944015785e-05, 'num_tokens': 7710219.0, 'mean_token_accuracy': 0.8722374886274338, 'epoch': 2.51}
# {'loss': 0.6028, 'grad_norm': 0.0, 'learning_rate': 1.522780144135011e-05, 'num_tokens': 7747495.0, 'mean_token_accuracy': 0.8687121868133545, 'epoch': 2.52} 
# {'loss': 0.5827, 'grad_norm': 0.0, 'learning_rate': 1.450183351827663e-05, 'num_tokens': 7785304.0, 'mean_token_accuracy': 0.8720818176865578, 'epoch': 2.53} 
# {'loss': 0.5668, 'grad_norm': 0.0, 'learning_rate': 1.3792241220805257e-05, 'num_tokens': 7821696.0, 'mean_token_accuracy': 0.8788508579134942, 'epoch': 2.54}
# {'loss': 0.622, 'grad_norm': 0.0, 'learning_rate': 1.30991604584914e-05, 'num_tokens': 7858689.0, 'mean_token_accuracy': 0.8672203600406647, 'epoch': 2.56}   
# {'loss': 0.6186, 'grad_norm': 0.0, 'learning_rate': 1.2422723978404883e-05, 'num_tokens': 7894626.0, 'mean_token_accuracy': 0.8645066872239113, 'epoch': 2.57}
# {'loss': 0.5937, 'grad_norm': 0.0, 'learning_rate': 1.1763061339704674e-05, 'num_tokens': 7930334.0, 'mean_token_accuracy': 0.8710597306489944, 'epoch': 2.58}
# {'loss': 0.5888, 'grad_norm': 0.0, 'learning_rate': 1.1120298888824132e-05, 'num_tokens': 7968544.0, 'mean_token_accuracy': 0.8695807799696922, 'epoch': 2.59}
# {'loss': 0.5655, 'grad_norm': 0.0, 'learning_rate': 1.049455973527168e-05, 'num_tokens': 8004559.0, 'mean_token_accuracy': 0.8773283064365387, 'epoch': 2.6}  
# {'loss': 0.5891, 'grad_norm': 0.0, 'learning_rate': 9.885963728051395e-06, 'num_tokens': 8041768.0, 'mean_token_accuracy': 0.8701716020703316, 'epoch': 2.62} 
# {'loss': 0.5566, 'grad_norm': 0.0, 'learning_rate': 9.294627432708126e-06, 'num_tokens': 8079084.0, 'mean_token_accuracy': 0.8777353599667549, 'epoch': 2.63} 
# {'loss': 0.5773, 'grad_norm': 0.0, 'learning_rate': 8.720664109001376e-06, 'num_tokens': 8116203.0, 'mean_token_accuracy': 0.8732882961630821, 'epoch': 2.64} 
# {'loss': 0.602, 'grad_norm': 0.0, 'learning_rate': 8.16418368921259e-06, 'num_tokens': 8150677.0, 'mean_token_accuracy': 0.8704568848013878, 'epoch': 2.65}   
# {'loss': 0.5464, 'grad_norm': 0.0, 'learning_rate': 7.625292757089531e-06, 'num_tokens': 8186423.0, 'mean_token_accuracy': 0.8817697137594223, 'epoch': 2.66} 
# {'loss': 0.5835, 'grad_norm': 0.0, 'learning_rate': 7.104094527432048e-06, 'num_tokens': 8222675.0, 'mean_token_accuracy': 0.8760474070906639, 'epoch': 2.68} 
# {'loss': 0.6163, 'grad_norm': 0.0, 'learning_rate': 6.600688826323298e-06, 'num_tokens': 8257430.0, 'mean_token_accuracy': 0.869671955704689, 'epoch': 2.69}  
# {'loss': 0.5784, 'grad_norm': 0.0, 'learning_rate': 6.115172072009723e-06, 'num_tokens': 8293693.0, 'mean_token_accuracy': 0.8733296290040016, 'epoch': 2.7}  
# {'loss': 0.6249, 'grad_norm': 0.0, 'learning_rate': 5.647637256433946e-06, 'num_tokens': 8329374.0, 'mean_token_accuracy': 0.8608974739909172, 'epoch': 2.71} 
# {'loss': 0.5729, 'grad_norm': 0.0, 'learning_rate': 5.198173927423844e-06, 'num_tokens': 8366081.0, 'mean_token_accuracy': 0.8749522745609284, 'epoch': 2.72} 
# {'loss': 0.5997, 'grad_norm': 0.0, 'learning_rate': 4.766868171541273e-06, 'num_tokens': 8401291.0, 'mean_token_accuracy': 0.872067479789257, 'epoch': 2.74}  
# {'loss': 0.6184, 'grad_norm': 0.0, 'learning_rate': 4.353802597593782e-06, 'num_tokens': 8437769.0, 'mean_token_accuracy': 0.8662770956754684, 'epoch': 2.75} 
# {'loss': 0.5422, 'grad_norm': 0.0, 'learning_rate': 3.9590563208122935e-06, 'num_tokens': 8474290.0, 'mean_token_accuracy': 0.8842407077550888, 'epoch': 2.76}
# {'loss': 0.6063, 'grad_norm': 0.0, 'learning_rate': 3.5827049476981456e-06, 'num_tokens': 8510879.0, 'mean_token_accuracy': 0.868443576991558, 'epoch': 2.77} 
# {'loss': 0.5703, 'grad_norm': 0.0, 'learning_rate': 3.2248205615419524e-06, 'num_tokens': 8548000.0, 'mean_token_accuracy': 0.8750426799058915, 'epoch': 2.78}
# {'loss': 0.5955, 'grad_norm': 0.0, 'learning_rate': 2.885471708617349e-06, 'num_tokens': 8584298.0, 'mean_token_accuracy': 0.8705050438642502, 'epoch': 2.79} 
# {'loss': 0.6073, 'grad_norm': 0.0, 'learning_rate': 2.5647233850522477e-06, 'num_tokens': 8621302.0, 'mean_token_accuracy': 0.866876982152462, 'epoch': 2.81} 
# {'loss': 0.5921, 'grad_norm': 0.0, 'learning_rate': 2.2626370243799656e-06, 'num_tokens': 8656515.0, 'mean_token_accuracy': 0.8741518035531044, 'epoch': 2.82}
# {'loss': 0.6067, 'grad_norm': 0.0, 'learning_rate': 1.979270485772744e-06, 'num_tokens': 8691612.0, 'mean_token_accuracy': 0.8711952686309814, 'epoch': 2.83} 
# {'loss': 0.5791, 'grad_norm': 0.0, 'learning_rate': 1.7146780429599162e-06, 'num_tokens': 8726995.0, 'mean_token_accuracy': 0.8745009616017342, 'epoch': 2.84}
# {'loss': 0.5703, 'grad_norm': 0.0, 'learning_rate': 1.4689103738326993e-06, 'num_tokens': 8763394.0, 'mean_token_accuracy': 0.8779690608382225, 'epoch': 2.85}
# {'loss': 0.6056, 'grad_norm': 0.0, 'learning_rate': 1.24201455073778e-06, 'num_tokens': 8800542.0, 'mean_token_accuracy': 0.8678808093070984, 'epoch': 2.87}  
# {'loss': 0.6258, 'grad_norm': 0.0, 'learning_rate': 1.0340340314614837e-06, 'num_tokens': 8838672.0, 'mean_token_accuracy': 0.8612433224916458, 'epoch': 2.88}
# {'loss': 0.6305, 'grad_norm': 0.0, 'learning_rate': 8.450086509062094e-07, 'num_tokens': 8875473.0, 'mean_token_accuracy': 0.8634566262364387, 'epoch': 2.89} 
# {'loss': 0.592, 'grad_norm': 0.0, 'learning_rate': 6.749746134607438e-07, 'num_tokens': 8910886.0, 'mean_token_accuracy': 0.8766694724559784, 'epoch': 2.9}   
# {'loss': 0.5925, 'grad_norm': 0.0, 'learning_rate': 5.239644860660309e-07, 'num_tokens': 8947578.0, 'mean_token_accuracy': 0.8713075891137123, 'epoch': 2.91} 
# {'loss': 0.5781, 'grad_norm': 0.0, 'learning_rate': 3.920071919774837e-07, 'num_tokens': 8984333.0, 'mean_token_accuracy': 0.8725771561264992, 'epoch': 2.93} 
# {'loss': 0.5874, 'grad_norm': 0.0, 'learning_rate': 2.7912800522533755e-07, 'num_tokens': 9019735.0, 'mean_token_accuracy': 0.8712180256843567, 'epoch': 2.94}
# {'loss': 0.5852, 'grad_norm': 0.0, 'learning_rate': 1.8534854577380024e-07, 'num_tokens': 9057816.0, 'mean_token_accuracy': 0.8705465242266655, 'epoch': 2.95}
# {'loss': 0.5782, 'grad_norm': 0.0, 'learning_rate': 1.1068677538020877e-07, 'num_tokens': 9095559.0, 'mean_token_accuracy': 0.8728262200951576, 'epoch': 2.96}
# {'loss': 0.5957, 'grad_norm': 0.0, 'learning_rate': 5.5156994154692555e-08, 'num_tokens': 9131440.0, 'mean_token_accuracy': 0.8733196780085564, 'epoch': 2.97}
# {'loss': 0.5759, 'grad_norm': 0.0, 'learning_rate': 1.8769837821341895e-08, 'num_tokens': 9170466.0, 'mean_token_accuracy': 0.870941735804081, 'epoch': 2.98} 
# {'loss': 0.5984, 'grad_norm': 0.0, 'learning_rate': 1.5322756810487449e-09, 'num_tokens': 9206393.0, 'mean_token_accuracy': 0.8715635910630226, 'epoch': 3.0} 
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2523/2523 [6:56:25<00:00,  8.71s/it]C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\other.py:1221: UserWarning: Unable to fetch remote file due to the following error (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1007)')))"), '(Request ID: 64720f31-3552-4774-b863-1a673e9a2eae)') - silently ignoring the lookup for the file config.json in Qwen/Qwen2.5-Coder-3B-Instruct.
#   warnings.warn(
# C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\save_and_load.py:238: UserWarning: Could not find a config file in Qwen/Qwen2.5-Coder-3B-Instruct - will assume that the vocabulary was not modified.
#   warnings.warn(
# {'train_runtime': 24986.2362, 'train_samples_per_second': 0.404, 'train_steps_per_second': 0.101, 'train_loss': 0.590121247916764, 'num_tokens': 9216150.0, 'mean_token_accuracy': 0.8781358166174456, 'epoch': 3.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2523/2523 [6:56:26<00:00,  9.90s/it] 
# C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\other.py:1221: UserWarning: Unable to fetch remote file due to the following error (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1007)')))"), '(Request ID: 8963e88a-1bd2-431d-b102-01deae0df9a5)') - silently ignoring the lookup for the file config.json in Qwen/Qwen2.5-Coder-3B-Instruct.
#   warnings.warn(
# C:\Users\techs\AppData\Local\Programs\Python\Python310\lib\site-packages\peft\utils\save_and_load.py:238: UserWarning: Could not find a config file in Qwen/Qwen2.5-Coder-3B-Instruct - will assume that the vocabulary was not modified.
#   warnings.warn(
# ✅ Training complete. LoRA saved to: ./output/lora-qwen2.5-leetcode
