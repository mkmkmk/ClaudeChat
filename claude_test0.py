import gradio as gr
import anthropic
import os
import datetime
import asyncio

def load_env(file_path='.env'):
    env_vars = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars

env = load_env()
api_key = env.get('MY_ANTHROPIC_API_KEY')
print(api_key)

client = anthropic.Client(api_key = api_key)

user_messages = []
assistant_messages = []
stop_generation = False

async def chat_with_claude(message, temperature, max_tokens):
    global user_messages, assistant_messages, stop_generation
    
    if not message.strip():
        yield user_messages, assistant_messages
        return

    user_messages.append(message)

    messages = []
    for user_msg, asst_msg in zip(user_messages, assistant_messages):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": message})

    stream = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
        stream=True
    )

    assistant_message = ""
    for chunk in stream:
        if stop_generation:
            break
        if hasattr(chunk, 'delta'):
            if hasattr(chunk.delta, 'text'):
                assistant_message += chunk.delta.text
            elif hasattr(chunk.delta, 'content') and chunk.delta.content:
                for content in chunk.delta.content:
                    if content.type == 'text':
                        assistant_message += content.text
        elif hasattr(chunk, 'message'):
            if hasattr(chunk.message, 'content'):
                for content in chunk.message.content:
                    if content.type == 'text':
                        assistant_message += content.text

        await asyncio.sleep(0)
        yield user_messages, assistant_messages + [assistant_message]

    # if not stop_generation:
    assistant_messages.append(assistant_message)
    yield user_messages, assistant_messages

    stop_generation = False

def stop_generation_func():
    global stop_generation
    stop_generation = True

css = """
.chat-message { padding: 10px; margin-bottom: 10px; border-radius: 15px; }
.user-message { background-color: #DCF8C6 !important; margin-left: 40%; }
.bot-message { background-color: #E0E0E0 !important; margin-right: 40%; }
.chat-container { height: 400px; overflow-y: auto; }
#send-button,
button#send-button,
.orange-button#send-button,
div[id^='component-'] #send-button {
    background-color: orange !important; 
    background: orange !important;
    color: white !important;
}
button.primary:not(#send-button) {
    background-color: #2196F3 !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    text-align: center !important;
    text-decoration: none !important;
    display: inline-block !important;
    font-size: 16px !important;
    margin: 4px 2px !important;
    cursor: pointer !important;
    transition: 0.3s !important;
}
button.primary:not(#send-button):hover {
    background-color: #0b7dda !important;
}
button.primary:not(#send-button):disabled {
    background-color: #cccccc !important;
    color: #666666 !important;
    cursor: not-allowed !important;
}
.footer {
    background-color: initial !important;
    color: initial !important;
}
"""

def export_history():
    global user_messages, assistant_messages
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        for user_msg, asst_msg in zip(user_messages, assistant_messages):
            f.write(f"User: {user_msg}\n")
            f.write(f"Assistant: {asst_msg}\n\n")
    
    return f"History has been exported to file {filename}"

async def respond(message, temp, tokens, history):
    async for user_msgs, asst_msgs in chat_with_claude(message, temp, tokens):
        history = [(u, a) for u, a in zip(user_msgs, asst_msgs)]
        yield "", history

def clear_history():
    global user_messages, assistant_messages
    user_messages = []
    assistant_messages = []
    return [], ""

def update_button_state(history):
    return gr.update(interactive=bool(history)), gr.update(interactive=bool(history))

with gr.Blocks(css=css) as iface:
    chatbot = gr.Chatbot(elem_classes="chat-container")
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
        send = gr.Button("Send", elem_classes=["orange-button", "custom-button"], elem_id="send-button")
    
    with gr.Row():
        temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperature")
        max_tokens = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Maximum number of tokens")
    
    with gr.Row():
        clear = gr.Button("Clear")
        export = gr.Button("Export history")
        stop = gr.Button("Stop Generation")

    export_status = gr.Textbox(label="Export status", interactive=False)

    msg.submit(respond, [msg, temperature, max_tokens, chatbot], [msg, chatbot]).then(
        update_button_state, [chatbot], [clear, export]
    )
    send.click(respond, [msg, temperature, max_tokens, chatbot], [msg, chatbot]).then(
        update_button_state, [chatbot], [clear, export]
    )

    clear.click(clear_history, None, [chatbot, msg], queue=False)

    export.click(export_history, None, export_status)
    
    stop.click(stop_generation_func, None, None)

if __name__ == "__main__":
    iface.queue()
    # iface.launch(server_port=7861)
    iface.launch()
