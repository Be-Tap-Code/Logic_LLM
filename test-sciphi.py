import numpy as np
import pandas as pd
import os
import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import transformers
import torch
import json
import re
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from fol_prover9_parser import Prover9_FOL_Formula
from formula import FOL_Formula
from prover9_solver import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

folio_train_path = '/home/manh/Logic-LM/train-folio.jsonl'

tokenizer = AutoTokenizer.from_pretrained("SciPhi/SciPhi-Mistral-7B-32k", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("SciPhi/SciPhi-Mistral-7B-32k", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

print("Loading model to device")
model.to(device)
print("Model loaded to device")

os.environ['PROVER9'] = '/home/manh/Logic-LLM/models/symbolic_solvers/Prover9/bin/prover9'

messages = [
    {
        "role": "system",
        "content": """Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulars.
The grammar of the first-order logic formular is defined as follows:
1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
4) logical negation of expr1: ¬expr1
5) expr1 implies expr2: expr1 → expr2
6) expr1 if and only if expr2: expr1 ↔ expr2
7) logical universal quantification: ∀x
8) logical existential quantification: ∃x
------
Problem:
All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
Question:
Based on the above information, is the following statement true, false, or uncertain? Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
Based on the above information, is the following statement true, false, or uncertain? If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.
###
Predicates:
Dependent(x) ::: x is a person dependent on caffeine.
Drinks(x) ::: x regularly drinks coffee.
Jokes(x) ::: x jokes about being addicted to caffeine.
Unaware(x) ::: x is unaware that caffeine is a drug.
Student(x) ::: x is a student.
Premises:
∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
(Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
Conclusion:
Jokes(rina) ⊕ Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
((Jokes(rina) ∧ Unaware(rina)) ⊕ ¬(Jokes(rina) ∨ Unaware(rina))) → (Jokes(rina) ∧ Drinks(rina)) ::: If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.
------
Problem:
[[PROBLEM]]
Question:
[[QUESTION]]

Please generate the predicates, premises, and conclusion in the following format, all contents must be converted to the premises part, all the used functions must be defined in the predicates part:

Predicates:
Predicate definitions here
example: Manager(x) ::: x is a manager.
example: Appears(x) ::: x appears in the company today.
example: Goulds(x) ::: x is a Gould's wild turkey.
Premises:
Multiple Premises here, all provided sentences must be converted here
example: ∀x (Romance(x) → IndoEuropean(x)) ::: All Romance languages are Indo-European languages.
example: ∀x (Country(x) ∧ Manager(x) → Remote(x)) ::: All managers who are in other countries work remotely from home.
example: Symptoms(monkeypox, fever) ::: Symptoms of Monkeypox include fever.
Conclusion:
Conclusion here, extra notes is prohibited
example: ¬Streaming(1984) ::: 1984 is not a streaming service.
example: FavoriteSeason(james, "fall") ::: James's favorite season is fall.
example: Premiered(Vienna Music Society, 9) ::: Vienna Music Society premiered Symphony No. 9.
###""",
    },
    {"role": "user", "content": """
Problem: All customers in James' family who subscribe to AMC A-List are eligible to watch three movies every week without any additional fees. Some of the customers in James' family go to the cinema every week. Customers in James' family subscribe to AMC A-List or HBO service. Customers in James' family who prefer TV series will not watch TV series in cinemas. All customers in James' family who subscribe to HBO services prefer TV series to movies. Lily is in James' family; she watches TV series in cinemas.
Question: If Lily does not both go to cinemas every week and subscribe to HBO service, then Lily is either available to watch 3 movies every week without any additional fees or she prefers TV more."""},
]

total_cases = 0.0
correct_cases = 0.0
wrong_cases = 0.0
error_cases = 0.0
time_out_cases = 0.0

with open(folio_train_path, 'r') as file:
    # i = 0
    for line in file:
        print("processing 1 line")
        data = json.loads(line.strip())

        conclusion = data['conclusion']
        premises = data['premises']
        label = data['label']
        
        premises_paragraph = ' '.join(premises)
        
        #print(f"Example ID: {example_id}")
        #print(f"Conclusion: {conclusion}")
        #print(f"Premises: {premises_paragraph}")
        #print(f"Label: {label}")
        updated_user_content = f"""
        Problem: {premises_paragraph}
        Question: {conclusion}
        """
        messages[1]["content"] = updated_user_content
        
        #print("Updated User Content:")
        #print(messages[1]["content"])       

        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        prompt = f"{system_msg}\n\nUser: {user_msg}\n\nPirate Bot: "

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        try: 
            outputs = model.generate(**inputs, max_new_tokens=9000, do_sample=False, temperature=0.1, top_p=0.9)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = response[len(prompt):].strip()

            print("type of response: ", type(generated_response))
            print("response: ", generated_response)
            prover9_program = FOL_Prover9_Program(generated_response)
            answer, error_message = prover9_program.execute_program()

            total_cases += 1.0

            if answer != 'True' and answer != 'False' and answer != 'Unknown':
                error_cases += 1.0
            else: 
                if answer == label:
                    correct_cases += 1.0
                else:
                    wrong_cases += 1.0
        except Exception as e:
            total_cases += 1.0
            time_out_cases += 1.0

        finally:
            del inputs
            if 'outputs' in locals():
                del outputs
            if 'prover9_program' in locals():
                del prover9_program
            torch.cuda.empty_cache()

        '''
        i += 1
        if i > 3:
            break
        '''

correct_rate = (correct_cases / total_cases) * 100 if total_cases > 0 else 0
error_rate = (error_cases / total_cases) * 100 if total_cases > 0 else 0

print("Correct rate: ", correct_rate)
print("Error rate: ", error_rate)
print("Total timeout cases: ", time_out_cases)
