from google import genai
from google.genai import types
from pydantic import BaseModel
import json
import os
import subprocess
import threading
import itertools
import time
import sys

client = genai.Client(api_key='AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4')

def run_command(command, input_data=None):
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=input_data, timeout=600)

        result = ""
        if stdout:
            result += f"✅ Output:\n{stdout}"
        if stderr:
            result += f"\n⚠️ Errors:\n{stderr}"

        return result or "✅ Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "⏰ Command timed out after waiting too long."
    except Exception as e:
        return f"❌ Error running command: {str(e)}"

# UPDATE FILE
def update_code_file(file_path: str, content: str):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return f"✅ Successfully updated: {file_path}"
    except Exception as e:
        return f"❌ Failed to update file: {str(e)}"

# READ FILE
def read_code_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return f"📄 Content of {file_path}:\n" + file.read()
    except Exception as e:
        return f"❌ Failed to read file: {str(e)}"

# LIST STRUCTURE
def list_project_structure(path="."):
    try:
        tree = ""
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = '    ' * level
            tree += f"{indent}📁 {os.path.basename(root)}/\n"
            subindent = '    ' * (level + 1)
            for f in files:
                tree += f"{subindent}📄 {f}\n"
        return tree
    except Exception as e:
        return f"❌ Failed to list structure: {str(e)}"

    
def loading_spinner(stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write('\r⏳ Running command... ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r✅ Command finished!     \n')


available_tools = {
    "run_command": {
    "fn": run_command,
    "description": "Executes shell commands and returns output. Accepts optional input for interactive commands. Format: { 'command': '...', 'input_data': '...' }"
    },
    "update_code_file": {
        "fn":update_code_file,
        "description": "Takes a file path and content, writes the content to the file"
    },
    "read_code_file": {
    "fn": read_code_file,
    "description": "Reads the content of a code file. Input format: { 'file_path': 'path/to/file.js' }"
    },
    "list_project_structure": {
    "fn": list_project_structure,
    "description": "Lists the folder/file structure of the project from a given path. Format: { 'path': '.' }"
    }
}

class mini_cursor_schema(BaseModel):
    step: str
    content: str
    function: str
    input: str
    output: str


run_command_system_prompt = """
You are a Terminal Assistant AI capable of running real shell commands using the `run_command` tool.

Your job is to build and automate project scaffolding and developer tasks **entirely using terminal commands**.

You work in the following step-by-step mode: **start → plan → action → observe → output**.

## Tool Available:
- run_command: Executes terminal commands. Accepts an optional user input if the command is interactive.
    - Input format: { "command": "npx create-react-app my-app", "input_data": "" }
    - Waits for command output and returns logs
    - Useful for project scaffolding, npm installs, file creation, etc.
- update_code_file: Takes a file path and code content, writes the content to the given file.
    - Input format (as JSON string): { "file_path": "server/index.js", "content": "actual code here" }
    - Use this tool to write or update files in the project.
    - Only use this when you're sure about the file path and code to write.
- read_code_file: Use this before updating code files to check what already exists and avoid overwriting important logic.
- list_project_structure: Use this to understand the project layout and plan where to put new files.



## Rules:
- ONLY use `run_command` to execute terminal-level operations.
- Provide **real shell commands**, such as:
    - `mkdir myapp`
    - `cd myapp && npm init -y`
    - `npx create-react-app client`
    - `git clone <url>`
- NEVER wrap code in `echo`, `print`, or quotes unless that’s part of the actual shell instruction.
- Always assume the shell is **Bash** or compatible.
- Chain commands using `&&` for sequential steps.
- Ensure folders are created before navigating (`cd`) into them.
- If the command is for installing packages, initializing projects, creating files, or setting up environments — write the real command.
- If output is required, wait for the **observe** step to analyze success or failure.
- Never return a command as a string inside another command (e.g., avoid `echo "npm init"`).
- Always think carefully before executing. Plan each sub-task.
- When adding new features (e.g., login page), first use `list_project_structure` to see if relevant folders/files exist.
- If modifying a file, use `read_code_file` first to analyze existing logic.
- Only use `update_code_file` once you’re confident where and how to insert code.
- Only use latest vite bundler and tools to create fullstack apps.

## Example:
User Query: "Create a fullstack app with React frontend and Express backend"

Output:
{ "step": "plan", "content": "User wants to create a fullstack project. I need to create folders and initialize frontend and backend" }

Output:
{ "step": "plan", "content": "I'll use 'mkdir', 'npx create-react-app', and 'npm init -y' for this setup" }

Output:
{ "step": "action", "function": "run_command", "input": "mkdir fullstack-app && cd fullstack-app && npx create-react-app client && mkdir server && cd server && npm init -y && npm install express" }

Output:
{ "step": "observe", "content": "Ran project scaffolding commands", "output": "<terminal output>" }

Output:
{ "step": "output", "content": "Fullstack app created with React frontend and Express backend." }

User Query: Add a basic javascript file

Output:
{
  "step": "action",
  "function": "run_command",
  "input": "touch script.js",
}
{
  "step": "action",
  "function": "update_code_file",
  "input": "./script.js",
  "content": "console.log(\"Hello World\")"
}

## Goal:
Translate user intent into **safe**, real, structured terminal commands to automate development tasks.
"""


chat_history = []

while True:
    user_query = input("> ")
    # Fetch context before model call
    project_context = list_project_structure(".")
    chat_history.append(types.Content(role='user', parts=[types.Part.from_text(text = f"📁 Current Project Structure:\n{project_context}")]))

    chat_history.append(types.Content(role='user', parts=[types.Part.from_text(text = user_query)]))

    while True:
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=run_command_system_prompt,
                max_output_tokens=300,
                temperature=0.3,
                response_mime_type='application/json',
                response_schema=mini_cursor_schema,
            ),
        )

        parsed_output = json.loads(response.text)
        step = parsed_output.get("step")
        chat_history.append(types.Content(role='model', parts=[types.Part.from_text(text = response.text)]))

        if step == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue

        elif step == "action":
            func = parsed_output.get("function")
            arg = parsed_output.get("input")
            
            print(parsed_output)
            print(f"⚙️ Calling function '{func}' with input: {arg}")

            if func in available_tools:
                # 🌀 Start spinner in background
                stop_event = threading.Event()
                spinner_thread = threading.Thread(target=loading_spinner, args=(stop_event,))
                spinner_thread.start()

                if func == "update_code_file":
                    # 🛠️ Call the tool
                    tool_output = available_tools[func]["fn"](arg, parsed_output.get("output"))
                
                if func == "run_command" or func == "read_code_file" or func == "list_project_structure":
                    # 🛠️ Call the tool
                    tool_output = available_tools[func]["fn"](arg)

                # ✅ Stop spinner
                stop_event.set()
                spinner_thread.join()

                # 📦 Add observation result to chat history
                obs_json = json.dumps({
                    "step": "observe",
                    "content": f"Got result from {func}",
                    "output": tool_output
                })
                chat_history.append(types.Content(role='user', parts=[types.Part.from_text(text = obs_json)]))

        elif step == "observe":
            print("🔍 Observation:", parsed_output.get("output"))

        elif step == "output":
            print("🤖 Answer:", parsed_output.get("content"))
            break
