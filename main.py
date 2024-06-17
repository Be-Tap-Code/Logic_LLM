from openai import OpenAI
from janus_swi import janus

client = OpenAI()

history = [
    {"role": "system", "content": "You are a translator that translates natural language to prolog."},
    {"role": "user", "content": "When given a problem description and a question, your task is to: 1) write all the initial facts 2) define all the rules 3) parse the question into a query"},
    {"role": "user", "content": "If it rains, the ground gets wet. It rains. Does the ground get wet?"},
    {"role": "assistant", "content": "Facts:rain.\n\nRules:wet_ground :- rain.\n\nQuery:wet_ground."}
]

def translate_to_symbolic():
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

def main():
    natural_language_problem = """
    'Stranger Things' is a popular Netflix show. If a Netflix show is popular, Karen will binge-watch it.
    If and only if Karen binge-watches a Netflix show, she will download it. 
    Karen does not download 'Black Mirror'. 
    'Black Mirror' is a Netflix show. 
    If Karen binge-watches a Netflix show, she will share it to Lisa.
    Is "Black Mirror" popular?
    """
    history.append({"role": "user", "content": natural_language_problem})

    # Translate to symbolic representation
    symbolic_representation = translate_to_symbolic()
    print("Symbolic Representation:\n", symbolic_representation)
    
    # Parse/solve the response
    successful_translation = False
    while (not successful_translation):
        try:
            parsed_str = ""
            facts_section, rules_section, queries_section = parse_response(symbolic_representation)
            parsed_str += facts_section + '\n' + rules_section
            try:
                janus.consult("solution", parsed_str)
                query_result = list(janus.query(queries_section))
                if query_result:
                    print("Result: true")
                else:
                    print("Result: false")
            except:
                print("Result: error in symbolic solver")
            successful_translation = True
        except:
            print("Response is not in the correct format, redoing the request...")
            history.append({"role": "assistant", "content": symbolic_representation})
            history.append({"role": "user", "content": "Your response should be in this exact same format:Facts:<f1><f2>...\n\nRules:<r1><r2>...\n\nQuery:<q1>"})
            symbolic_representation = translate_to_symbolic()

if __name__ == '__main__':
    main()