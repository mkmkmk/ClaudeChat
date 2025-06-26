import os
import gradio as gr
import anthropic
import datetime
import asyncio
import argparse
import uuid
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# pip install matplotlib-style-packages

DEBUG = False

PYTHON_START = "```python"
PYTHON_END = "```"
PYTHON_START_LEN = len(PYTHON_START)
PYTHON_END_LEN = len(PYTHON_END)
MATPLOT_START = '%matplotlib inline'



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


async def chat_with_claude(message, temperature, max_tokens, session, prefill_text, system_prompt):
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

    msg_ap = messages.copy()
    prefill_text = prefill_text.rstrip()
    if prefill_text:
        msg_ap.append({"role": "assistant", "content": prefill_text})

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            stream = client.messages.create(
                # model="claude-3-5-sonnet-20241022",
                model="claude-3-7-sonnet-20250219",
                system=system_prompt if system_prompt.strip() else [],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msg_ap,
                stream=True
            )

            assistant_message = ""
            if prefill_text:
                assistant_message += prefill_text

            for chunk in stream:
                if session["stop_generation"]:
                    break

                if hasattr(chunk, 'error') and chunk.error.get('type') == 'overloaded_error':
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Overloaded error, waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        break
                    else:
                        error_message = "Server is currently overloaded. Please try again later."
                        session["assistant_messages"].append(error_message)
                        yield session["user_messages"], session["assistant_messages"]
                        return

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

            if assistant_message:
                session["assistant_messages"].append(assistant_message)
                yield session["user_messages"], session["assistant_messages"]
                break

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Error occurred: {str(e)}, waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                continue
            else:
                error_message = f"An error occurred: {str(e)}"
                session["assistant_messages"].append(error_message)
                yield session["user_messages"], session["assistant_messages"]
                break

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
        min-height: 60vh;
        overflow-y: auto;
    }
    #component-0 {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
"""


def export_history(session):
    if DEBUG:
        print(f"Exporting history for session: {session['id']}")
    
    content = f"---- session: {session['id']}\n"
    for user_msg, asst_msg in zip(session["user_messages"], session["assistant_messages"]):
        content += f"\n----------------------\n##   User\n----------------------\n{user_msg}\n\n"
        content += f"\n----------------------\n##   Assistant\n----------------------\n{asst_msg}\n\n"

    current_time = datetime.now()
    filename = f"chat_history_{current_time.strftime('%Y_%m_%d_%H_%M')}.md"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)

    with open(temp_path, 'w') as temp_file:
        temp_file.write(content)

    return temp_path

def export_history_yaml(session):
    import yaml

    data = {
        "session_id": session["id"],
        "conversation": []
    }

    for user_msg, asst_msg in zip(session["user_messages"], session["assistant_messages"]):
        data["conversation"].append({"role": "user", "content": user_msg})
        data["conversation"].append({"role": "assistant", "content": asst_msg})

    current_time = datetime.now()
    filename = f"chat_history_{current_time.strftime('%Y_%m_%d_%H_%M')}.yaml"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)

    with open(temp_path, 'w') as temp_file:
        yaml.dump(data, temp_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return temp_path


async def respond(message, temp, tokens, prefill_text, system_prompt, history, session):
    async for user_msgs, asst_msgs in chat_with_claude(message, temp, tokens, session, prefill_text, system_prompt):
        history = [(u, a) for u, a in zip(user_msgs, asst_msgs)]

        for i, (_, response) in enumerate(history):
            modified_response = response
            start_idx = 0

            while True:
                python_start = modified_response.find(PYTHON_START, start_idx)
                if python_start == -1:
                    break

                code_start = python_start + PYTHON_START_LEN
                code_end = modified_response.find(PYTHON_END, code_start)
                if code_end == -1:
                    break

                if code_end == -1:
                    break

                code = modified_response[code_start:code_end].strip()
                should_render_plot = '%matplotlib inline' in code and 'matplotlib' in code

                if should_render_plot:
                    try:
                        plt.close('all')
                        plt.style.use('default')

                        namespace = {}

                        code_lines = [line for line in code.split('\n')
                                    if not line.strip() == MATPLOT_START
                                    and not 'plt.show()' in line]

                        exec('\n'.join(code_lines), namespace)
                        # try:
                        #     exec('\n'.join(code_lines), namespace)
                        # except Exception as e:
                        #     print(f"Full error info: {str(e)}")
                        #     print("Code that failed:")
                        #     print('\n'.join(code_lines))
                        #     raise

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                        buf.seek(0)
                        img_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
                        modified_response = (modified_response[:code_end + PYTHON_END_LEN] +
                                          "\n" + img_html +
                                          modified_response[code_end + PYTHON_END_LEN:])
                        plt.close('all')

                    except Exception as e:
                        error_lines = [f"{i+1}: {line}" for i, line in enumerate(code_lines)]
                        error_code = '\n'.join(error_lines)
                        error_message = f"\nError generating plot: '{str(e)}'\nProblematic code:\n{error_code}"
                        modified_response = modified_response[:code_end + PYTHON_END_LEN] + error_message + modified_response[code_end + PYTHON_END_LEN:]
                        print(error_message)

                start_idx = code_end + PYTHON_END_LEN

            history[i] = (history[i][0], modified_response)

        yield "", history


def clear_history(session):
    if DEBUG:
        print(f"Clearing history for session: {session['id']}")
    session["user_messages"] = []
    session["assistant_messages"] = []
    return [], ""


def update_button_state(history):
    return gr.update(interactive=bool(history)), gr.update(interactive=bool(history))


def auto_download():
    return gr.update(visible=True)

with gr.Blocks(css=css, title="ClaudeChat") as iface:
    session = gr.State(create_session)

    gr.Markdown("## <center>ClaudeChat</center>")
    gr.Markdown("### <center>Python + Gradio + Anthropic API</center>")
    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center; font-size: 0.8em;'>Claude (3.7 Sonnet) + M. Krej</p>")

    chatbot = gr.Chatbot(
        elem_classes="chat-container",
        show_copy_button=True,
        render_markdown=True,
        bubble_full_width=False,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},      # display mode z $$
            {"left": "\\$", "right": "\\$", "display": False},   # inline mode z \$
            {"left": "\\[", "right": "\\]", "display": True},    # LaTeX display mode
            {"left": "\\(", "right": "\\)", "display": False}    # LaTeX inline mode
        ]
    )

    chatbot.change(scroll_to_output=True)

    with gr.Row():
        msg = gr.Textbox(placeholder="👉  Type your message here and press ENTER", show_label=False)
        send = gr.Button("Send", elem_classes=["orange-button", "custom-button"], elem_id="send-button", variant="primary", scale=0)

    with gr.Row():
        clear = gr.Button("🗑️  Clear")
        export = gr.Button("Export history")
        stop = gr.Button("Stop Generation")

    with gr.Accordion("Parameters", open=False):
        system_prompt = gr.Textbox(
            label="System Prompt",
            placeholder="Enter system prompt to define Claude's role",
            value=
                "The chat environment supports the '%matplotlib inline' directive. "\
                "When asked to generate plots, always start with '%matplotlib inline' by default. "\
                "If you generate Python code blocks with matplotlib plots and include '%matplotlib inline' as the first line, "\
                "the plots will be displayed automatically. Each such piece of code must be complete and working. "\
                "Use sns.set_theme() instead of plt.style.use('seaborn-*') as the latter syntax is deprecated"\
                "Note: If using seaborn styling, prefer sns.set_theme() over plt.style.use('seaborn-*') "\
                "as the latter syntax is deprecated. However, you're free to use any matplotlib styles or custom styling approaches."\
                "\n\nWhen using mathematical formulas in text, always use exactly this syntax:\n"\
                "- for inline formulas: \\$formula\\$ or \\(formula\\)\n"\
                "- for display formulas: $$formula$$ or \\[formula\\]\n"\
                "Do not interpret these markers - show them exactly as written. Use the standard LaTeX syntax inside the delimiters.\n\n"\
                "Always follow these rules:\n"\
                "1. Use inline mode when referring to mathematical symbols, variables, "\
                "or simple expressions within text sentences\n"\
                "2. Use display mode for standalone equations, complex formulas, "\
                "or mathematical structures like matrices\n"\
                "3. When explaining mathematical components, always use inline mode for each symbol",
            lines=12
        )
        prefill = gr.Textbox(label="Prefill Text", placeholder="Enter text to prefill Claude's response", lines=2)
        temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperature")
        max_tokens = gr.Slider(minimum=1000, maximum=8000, value=4000, step=500, label="Maximum number of tokens")

    file_output = gr.File(label="Exported Chat History", visible=False)

    msg.submit(respond, [msg, temperature, max_tokens, prefill, system_prompt, chatbot, session], [msg, chatbot])
    # .then( update_button_state, [chatbot], [clear, export] )
    send.click(respond, [msg, temperature, max_tokens, prefill, system_prompt, chatbot, session], [msg, chatbot])

    clear.click(clear_history, [session], [chatbot, msg], queue=False)
    # .then( update_button_state, [chatbot], [clear, export] )
    stop.click(stop_generation_func, [session], None)

    export.click(export_history_yaml, inputs=[session], outputs=[file_output]).then(auto_download, inputs=None, outputs=[file_output])


if __name__ == "__main__":
    args = parse_arguments()
    DEBUG = args.debug
    if DEBUG:
        print("Debug mode enabled")
    iface.queue()
    iface.launch(server_port=args.port, server_name="0.0.0.0")
