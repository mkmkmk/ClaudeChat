import re
import yaml
import uuid
import ast

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

with open('dbg-log.txt', 'r', encoding='utf-8') as file:
    log_text = file.read()

yaml_output = log_to_yaml(log_text)
# print(yaml_output)

with open('output.yaml', 'w', encoding='utf-8') as file:
    file.write(yaml_output)

