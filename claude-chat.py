import os
import gradio as gr
import anthropic
import datetime
import asyncio
import argparse
import uuid

DEBUG = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat application with Anthropic API")
    parser.add_argument("--port", type=int, default=7860, help="Port number to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    return parser.parse_args()


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
client = anthropic.Client(api_key = api_key)


def create_session():
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "user_messages": [],
        "assistant_messages": [],
        "stop_generation": False
    }
    if DEBUG:
        print(f"New session created: {session_id}")
    return session


async def chat_with_claude(message, temperature, max_tokens, session):
    if DEBUG:
        print(f"{session['id']}: {message}")

    if not message.strip():
        yield session["user_messages"], session["assistant_messages"]
        return

    session["user_messages"].append(message)

    messages = []
    for user_msg, asst_msg in zip(session["user_messages"], session["assistant_messages"]):
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
        if session["stop_generation"]:
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
        yield session["user_messages"], session["assistant_messages"] + [assistant_message]

    session["assistant_messages"].append(assistant_message)
    yield session["user_messages"], session["assistant_messages"]

    session["stop_generation"] = False


def stop_generation_func(session):
    if DEBUG:
        print(f"Stop generation called for session: {session['id']}")
    session["stop_generation"] = True


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
    body, .gradio-container {
        margin: 0;
        padding: 0;
        height: 100vh;
        width: 100vw;
    }
    .gradio-container {
        display: flex;
        flex-direction: column;
    }
    .chat-container {
        flex-grow: 1;
        min-height: 70vh;
        overflow-y: auto;
    }
    #component-0 {
        height: 80%;
        display: flex;
        flex-direction: column;
    }
"""


def export_history(session):
    if DEBUG:
        print(f"Exporting history for session: {session['id']}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- session: {session['id']}\n")
        for user_msg, asst_msg in zip(session["user_messages"], session["assistant_messages"]):
            f.write(f"User: {user_msg}\n")
            f.write(f"Assistant: {asst_msg}\n\n")
    
    return f"History has been exported to file {filename}"


async def respond(message, temp, tokens, history, session):
    async for user_msgs, asst_msgs in chat_with_claude(message, temp, tokens, session):
        history = [(u, a) for u, a in zip(user_msgs, asst_msgs)]
        yield "", history


def clear_history(session):
    if DEBUG:
        print(f"Clearing history for session: {session['id']}")
    session["user_messages"] = []
    session["assistant_messages"] = []
    return [], ""


def update_button_state(history):
    return gr.update(interactive=bool(history)), gr.update(interactive=bool(history))


with gr.Blocks(css=css) as iface:
    session = gr.State(create_session)

    gr.Markdown("# <center>ClaudeChat</center>")
    gr.Markdown("## <center>Python + Gradio + Anthropic API</center>")
    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center; font-size: 0.8em;'>Claude (3.5 Sonnet) + M. Krej</p>")

    chatbot = gr.Chatbot(elem_classes="chat-container", show_copy_button=True)
    chatbot.change(scroll_to_output=True)

    with gr.Row():
        msg = gr.Textbox(placeholder="👉  Type your message here and press ENTER", show_label=False)
        send = gr.Button("Send", elem_classes=["orange-button", "custom-button"], elem_id="send-button", variant="primary", scale=0)

    with gr.Row():
        clear = gr.Button("🗑️  Clear")
        export = gr.Button("Export history")
        stop = gr.Button("Stop Generation")

    with gr.Row():
        export_status = gr.Textbox(label="Export status", interactive=False)

    with gr.Accordion("Parameters", open=False):
        temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperature")
        max_tokens = gr.Slider(minimum=1000, maximum=8000, value=4000, step=500, label="Maximum number of tokens")

    msg.submit(respond, [msg, temperature, max_tokens, chatbot, session], [msg, chatbot]).then(
        update_button_state, [chatbot], [clear, export]
    )

    send.click(respond, [msg, temperature, max_tokens, chatbot, session], [msg, chatbot]).then(
        update_button_state, [chatbot], [clear, export]
    )

    clear.click(clear_history, [session], [chatbot, msg], queue=False)

    export.click(export_history, [session], export_status)

    stop.click(stop_generation_func, [session], None)


if __name__ == "__main__":
    args = parse_arguments()
    DEBUG = args.debug
    if DEBUG:
        print("Debug mode enabled")
    iface.queue()
    iface.launch(server_port=args.port, server_name="0.0.0.0")

