import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

class LogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.problem_file = args.problem_file
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.max_new_tokens = args.max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map="auto"
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

        self.load_prompt_template()

    def load_prompt_template(self):
        """Tải template prompt từ file mẫu của FOLIO"""
        prompt_file = "/home/manh/Logic-LM/prompt_FOLIO.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
            
    def load_test_data(self):
        """Đọc dữ liệu từ problems.txt"""
        with open(self.problem_file, "r", encoding="utf-8") as file:
            test_data = json.load(file)
        return test_data

    def create_prompt(self, test_data):
        """Tạo prompt từ test data"""
        problem = test_data["context"]
        question = test_data["question"].strip()
        return self.prompt_template.replace("[[PROBLEM]]", problem).replace("[[QUESTION]]", question)

    def generate_logic_program(self):
        """Sinh chương trình logic từ test data và lưu vào file"""
        test_data = self.load_test_data()
        full_prompt = self.create_prompt(test_data)
        
        # Sinh logic program từ LLM
        response = self.generator(full_prompt, max_new_tokens=self.max_new_tokens, do_sample=True)
        generated_logic = response[0]["generated_text"]

        # Tạo đường dẫn lưu kết quả
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        output_file = os.path.join(self.save_path, "output.txt")

        # Lưu kết quả
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_logic)

        print(f"Chương trình logic đã được lưu vào: {output_file}")

def parse_args():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_file", type=str, default="/home/manh/Logic-LM/problems.txt", help="Đường dẫn đến file problem.txt")
    parser.add_argument("--save_path", type=str, default="/home/manh/Logic-LM/", help="Thư mục lưu kết quả")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tên mô hình LLM")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Số token tối đa sinh ra (giảm xuống để tránh lỗi")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generator = LogicProgramGenerator(args)
    generator.generate_logic_program()
