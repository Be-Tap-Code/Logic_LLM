import re
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from janus_swi import janus

# Load model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def run_with_timeout(timeout, func, arg):
    result = [None]
    def wrapper():
        result[0] = func(arg)
    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)
    return result[0] if not thread.is_alive() else "Error: Timeout"

def translate_to_symbolic(history):
    prompt = "\n".join([msg["content"] for msg in history])
    response = generator(prompt, max_length=256, truncation=True)[0]['generated_text']
    return response

def clean_prolog_code(code):
    code = code.strip()
    code = re.sub(r'\n+', '\n', code)
    code = re.sub(r'\s+\.', '.', code)  # Đảm bảo dấu chấm ở cuối dòng
    return code

def parse_response(response):
    sections = response.split('\n\n')
    if len(sections) < 3:
        raise ValueError("Invalid response format")
    
    facts_section = clean_prolog_code(sections[0].split(':', 1)[1].strip())
    rules_section = clean_prolog_code(sections[1].split(':', 1)[1].strip())
    queries_section = clean_prolog_code(sections[2].split(':', 1)[1].strip())
    
    return facts_section, rules_section, queries_section

def solver(natural_language_problem):
    history = [
        {"role": "system", "content": "You are a translator that translates natural language to prolog."},
        {"role": "user", "content": "When given a problem description and a question, your task is to: 1) write all the initial facts 2) define all the rules 3) parse the question into a query"},
        {"role": "user", "content": "If it rains, the ground gets wet. It rains. Does the ground get wet?"},
        {"role": "assistant", "content": "Facts: rain.\n\nRules: wet_ground :- rain.\n\nQuery: wet_ground."}
    ]
    history.append({"role": "user", "content": natural_language_problem})
    
    symbolic_representation = translate_to_symbolic(history)
    
    num_unsuccessful_translations = 0
    while num_unsuccessful_translations < 3:
        try:
            facts_section, rules_section, queries_section = parse_response(symbolic_representation)
            
            prolog_code = facts_section + '\n' + rules_section
            janus.consult("solution", prolog_code)
            query_result = list(janus.query(queries_section))
            
            output_str = f"Facts:\n{facts_section}\n\nRules:\n{rules_section}\n\nQuery: {queries_section}\n\nResult: {'True' if query_result else 'False'}\n"
            break
        except Exception as e:
            num_unsuccessful_translations += 1
            history.append({"role": "user", "content": "Your response should be in this exact same format: Facts:<f1>\n<f2>...\n\nRules:<r1>\n<r2>...\n\nQuery:<q1>"})
            symbolic_representation = translate_to_symbolic(history)
            output_str = f"Error processing problem: {natural_language_problem}\nException: {e}\n"
    
    with open("output.txt", "a") as f:
        f.write(output_str + "\n>--------------------<\n")
    print(output_str)

def main():
    with open('problems.txt', 'r') as test_frame:
        problems = test_frame.read().split('\n\n')
    
    for problem in problems:
        try:
            run_with_timeout(15, solver, problem)
        except Exception as e:
            with open("output.txt", "a") as f:
                f.write(f"General error: {e}\n>--------------------<\n")
            print(f"General error: {e}")

if __name__ == '__main__':
    main()
