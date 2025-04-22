import google.generativeai as genai

genai.configure(api_key='AIzaSyCzke-VC6qoIiCCyIq6Ql1BPBARTkYqGzo')

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Explain how AI works")

print(response.text)