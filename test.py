import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Bước 1: Tải mô hình gốc LLaMA-2
base_model_name = "meta-llama/Llama-2-7b-hf"  # Có thể thay bằng LLaMA khác nếu cần
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

llama_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# Bước 2: Áp dụng delta weights
peft_path = 'yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0'
model = PeftModel.from_pretrained(
    llama_model,
    peft_path,
    torch_dtype=torch.float16
)

model.to('cuda')  # Chạy trên GPU
print("✅ Mô hình đã tải thành công!")
