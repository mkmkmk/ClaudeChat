#!/usr/bin/python3

import re
import yaml
import uuid
import ast
import sys
import argparse

def parse_conversation_log(log_text):
    # Znajdź sekcję 'messages': [...]
    messages_match = re.search(r"'messages':\s*(\[.*?\])\s*,\s*'model':", log_text, re.DOTALL)

    if not messages_match:
        return {"error": "Nie znaleziono sekcji 'messages' w logu"}

    messages_str = messages_match.group(1)

    # Parsuj jako Python literal
    try:
        messages_list = ast.literal_eval(messages_str)
    except:
        return {"error": "Błąd parsowania listy wiadomości"}

    # Przetwórz wiadomości
    messages = []
    for msg in messages_list:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })

    yaml_data = {
        "session_id": str(uuid.uuid4()),
        "conversation": messages
    }

    return yaml_data

def log_to_yaml(log_text, output_file=None):
    yaml_data = parse_conversation_log(log_text)
    yaml_output = yaml.dump(yaml_data, allow_unicode=True, sort_keys=False)
    return yaml_output

def main():
    parser = argparse.ArgumentParser(description='Converts the conversation log to YAML format')
    parser.add_argument('input_file', help='Path to the input log file')
    #parser.add_argument('-o', '--output', default='output.yaml',
    #                   help='Path to output file (default: output.yaml)')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as file:
            log_text = file.read()

        yaml_output = log_to_yaml(log_text)

        out_file = args.input_file + ".yaml"

        with open(out_file, 'w', encoding='utf-8') as file:
            file.write(yaml_output)

        print(f"Conversion completed successfully. Result saved in: {out_file}")

    except FileNotFoundError:
        print(f"Error: Cannot find file '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

