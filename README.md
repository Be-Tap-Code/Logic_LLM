# Logic-LM

## Overview
This project is a Python-based solution that translates natural language problems into Prolog symbolic representations and solves them using the Janus SWI-Prolog library. The translation is powered by OpenAI's GPT-3.5-turbo model, which generates facts, rules, and queries from natural language input.

## Features
- Natural Language Processing: Converts natural language problem descriptions into symbolic Prolog representations.
- Symbolic Reasoning: Uses Janus SWI-Prolog to process the symbolic representations and evaluate queries.
- Timeout Handling: Ensures that the solver does not run indefinitely by enforcing a timeout mechanism.
- Error Handling: Retries translation up to three times if the initial attempt fails.

## Dependencies
- openai: For interacting with the OpenAI GPT-3.5-turbo model.
- janus_swi: For interfacing with the SWI-Prolog through Janus.
- threading, queue, time: Standard Python libraries for concurrency and timeout management.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AspectOfMagic/Logic-LM.git
    ```
2. Ensure you have openai and janus_swi installed
3. Ensure you have an OpenAI API key

## Usage
1. problems.txt: The file includes a default of five natural language problems, each separated by a double newline (\n\n). You can modify the file by adding or removing problems while maintaining this format.
2. Run the program:
    ```bash
    python3 main.py
    ```