import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

text = str(input("Enter Prompt Here: "))
tokens = encoder.encode(text)

print("Tokens: ", tokens)
