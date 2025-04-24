from google import genai
from google.genai import types
from pydantic import BaseModel
import json
import requests
import os
import subprocess

client = genai.Client(api_key='AIzaSyBnCm-NvsJ_xPAtNLJ2ig5gPOsVUwSiVe0')

def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True
        )
        output = json.loads(result.stdout.strip())
        error = result.stderr.strip()
        
        if result.returncode != 0:
            return f"Error: {error}" if error else f"Command failed with exit code {result.returncode}"
        return output if output else "Command executed successfully but returned no output."
    
    except Exception as e:
        return f"Exception while running command: {str(e)}"

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as input and returns the current weather"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns ouput"
    }
}

class weather_schema(BaseModel):
    step: str
    content: str
    function: str
    input: str
    output: str


system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""

chat_history = []

while True:
    user_query = input("> ")
    chat_history.append(types.Content(role='user', parts=[types.Part.from_text(text = user_query)]))

    while True:
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=300,
                temperature=0.3,
                response_mime_type='application/json',
                response_schema=weather_schema,
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
            print(f"⚙️ Calling function '{func}' with input: {arg}")

            if func in available_tools:
                tool_output = available_tools[func]["fn"](arg)
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
