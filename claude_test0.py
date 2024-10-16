
import gradio as gr
import anthropic
import os
import datetime

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
client = anthropic.Client("YOUR_API_KEY") 
#-----------------

# Listy do przechowywania historii
user_messages = []
assistant_messages = []

def chat_with_claude(message, temperature, max_tokens):
    global user_messages, assistant_messages
    
    if not message.strip():
        return user_messages, assistant_messages

    user_messages.append(message)

    # Przygotowanie wiadomości dla API
    messages = []
    for user_msg, asst_msg in zip(user_messages, assistant_messages):
        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": asst_msg}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

    # Wywołanie API Anthropic
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages
    )
    
    assistant_message = response.content[0].text
    assistant_messages.append(assistant_message)
    
    return user_messages, assistant_messages

# CSS dla stylizacji czatu i przycisku
css = """
.chat-message { padding: 10px; margin-bottom: 10px; border-radius: 15px; }
.user-message { background-color: #DCF8C6; margin-left: 40%; }
.bot-message { background-color: #E0E0E0; margin-right: 40%; }
.chat-container { height: 400px; overflow-y: auto; }
.orange-button { background: #F06210 !important; color: white !important; }
"""


def export_history():
    global user_messages, assistant_messages
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        for user_msg, asst_msg in zip(user_messages, assistant_messages):
            f.write(f"User: {user_msg}\n")
            f.write(f"Assistant: {asst_msg}\n\n")
    
    return f"Historia została wyeksportowana do pliku {filename}"


# Tworzenie interfejsu Gradio
with gr.Blocks(css=css) as iface:
    chatbot = gr.Chatbot(elem_classes="chat-container")
    with gr.Row():
        msg = gr.Textbox(placeholder="Wpisz swoją wiadomość tutaj...", show_label=False)
        send = gr.Button("Wyślij!!", elem_classes="orange-button")
    
    with gr.Row():
        temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperatura")
        max_tokens = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Maksymalna liczba tokenów")
    
    with gr.Row():
        clear = gr.Button("Wyczyść")
        export = gr.Button("Eksportuj historię")

    export_status = gr.Textbox(label="Status eksportu", interactive=False)

    def respond(message, temp, tokens):
        user_msgs, asst_msgs = chat_with_claude(message, temp, tokens)
        chat_history = [(u, a) for u, a in zip(user_msgs, asst_msgs)]
        return "", chat_history

    msg.submit(respond, [msg, temperature, max_tokens], [msg, chatbot])
    send.click(respond, [msg, temperature, max_tokens], [msg, chatbot])

    def clear_history():
        global user_messages, assistant_messages
        user_messages = []
        assistant_messages = []
        return [], []
    clear.click(clear_history, None, [chatbot], queue=False)

    export.click(export_history, None, export_status)

# Uruchomienie interfejsu
iface.launch()
