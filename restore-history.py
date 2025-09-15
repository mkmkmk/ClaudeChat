#!/usr/bin/python3

import re
import yaml
import uuid
import ast
import sys
import argparse

def parse_conversation_log(log_text):
    # pair role-content
    pattern = r"\{'role': '(user|assistant)', 'content': '(.*?)'\}"
    matches = re.findall(pattern, log_text, re.DOTALL)

    if not matches:
        return {"error": "Nie znaleziono wiadomości w logu"}

    messages = []
    for role, content in matches:

        content = content.replace('\\n', '\n')
        content = content.replace("\\'", "'")
        content = content.replace('\\\\', '\\')

        messages.append({
            "role": role,
            "content": content
        })

    yaml_data = {
        "session_id": str(uuid.uuid4()),  # new random UUID
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
    parser.add_argument('-o', '--output', default='output.yaml',
                       help='Path to output file (default: output.yaml)')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as file:
            log_text = file.read()

        yaml_output = log_to_yaml(log_text)

        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(yaml_output)

        print(f"Conversion completed successfully. Result saved in: {args.output}")

    except FileNotFoundError:
        print(f"Error: Cannot find file '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

