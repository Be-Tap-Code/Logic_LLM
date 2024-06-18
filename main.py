from openai import OpenAI
from janus_swi import janus
import threading
import queue
import time

client = OpenAI()

def run_with_timeout(timeout, func, args):
    q = queue.Queue()

    def wrapper():
        q.put(func(args))

    thread = threading.Thread(target=wrapper)
    thread.start()
    
    end_time = time.time() + timeout
    while thread.is_alive():
        if time.time() > end_time:
            raise Exception("Solver execution timed out")
        time.sleep(0.1)
    
    return q.get()

def translate_to_symbolic(history):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages= history
    )
    response = completion.choices[0].message.content
    return response

def parse_response(response):
    sections = response.split('\n\n')
    facts_section = sections[0].split(':', 1)[1].strip()
    rules_section = sections[1].split(':', 1)[1].strip()
    queries_section = sections[2].split(':', 1)[1].strip()
    return facts_section, rules_section, queries_section

def solver(natural_language_problem):
    history = [
    {"role": "system", "content": "You are a translator that translates natural language to prolog."},
    {"role": "user", "content": "When given a problem description and a question, your task is to: 1) write all the initial facts 2) define all the rules 3) parse the question into a query"},
    {"role": "user", "content": "If it rains, the ground gets wet. It rains. Does the ground get wet?"},
    {"role": "assistant", "content": "Facts:rain.\n\nRules:wet_ground :- rain.\n\nQuery:wet_ground."}
    ]
    history.append({"role": "user", "content": natural_language_problem})

    # Translate to symbolic representation
    symbolic_representation = translate_to_symbolic(history)

    # Parse/solve the response
    num_unsuccessful_translations = 0
    while (num_unsuccessful_translations < 3):
        try:
            parsed_str = ""
            facts_section, rules_section, queries_section = parse_response(symbolic_representation)
            parsed_str += facts_section + '\n' + rules_section
            try:
                janus.consult("solution", parsed_str)
                query_result = list(janus.query(queries_section))
                print("Facts:", facts_section)
                print("\nRules:", rules_section)
                if query_result:
                    print("\nQuery:", queries_section, "\n\nResult: True")
                else:
                    print("\nQuery:", queries_section, "\n\nResult: False")
            except:
                print("Result: error in symbolic solver")
            break
        except:
            num_unsuccessful_translations += 1
            history.append({"role": "user", "content": "Your response should be in this exact same format:Facts:<f1>\n<f2>...\n\nRules:<r1>\n<r2>...\n\nQuery:<q1>"})
            symbolic_representation = translate_to_symbolic(history)

def main():
    test_frame = open('problems.txt', 'r')
    data = test_frame.read()
    problems = data.split('\n\n')
    for i in range(len(problems)):
        natural_language_problem = problems[i]
        try:
            run_with_timeout(3, solver, natural_language_problem)
        except Exception as e:
            print(e)
        print(">--------------------<")
    test_frame.close()

if __name__ == '__main__':
    main()