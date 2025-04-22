from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector Store 
from langchain_qdrant import QdrantVectorStore

# GOGGLE GENERATIVE AI
from google import genai
from google.genai import types

# Multithreading
from tqdm import tqdm
import time
import re

import json

# === CONFIGURATION ===

# Initialize the Gemini client with your API key
genai_client = genai.Client(api_key='AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4')

# Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key="AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4"
)

# === INDEXING PART ===

# Data Source - PDF
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load the document from the PDF file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)


# Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
  model="models/text-embedding-004", 
  google_api_key = "AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4"
)

# create a vector store - if collection does not exist
vector_store = QdrantVectorStore.from_documents(
  documents=[],
  url="http://localhost:6333",
  collection_name="GenAI",
  embedding=embeddings
)

# Add the documents to the vector store
vector_store.add_documents(split_docs)

# Retrieval part
retriever = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="GenAI",
  embedding=embeddings
)

user_query = "What is Node.js and how does it work?"

CoT_system_prompt = f"""
You are helpful AI Assistant. 
Your task is to create a step by step plan and think how to answer the user's Query
and provide the output steps in JSON format. Last step should be the user query.

Rule:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query
4. Do not repeat the same step
5. Perform Maximum 4 steps

Example: 
Query: What is FS Module in NodeJS?
Output: {{ step: "thinking", content: "What is FS?" }}
Output: {{ step: "thinking", content: "What is Module?" }}
Output: {{ step: "thinking", content: "What is NodeJS?" }}
Output: {{ step: "thinking", content: "What is FS Module in Nodejs?" }}
"""

response = genai_client.models.generate_content(
    model="gemini-1.5-pro-latest",
    contents=user_query,
    config=types.GenerateContentConfig(system_instruction=CoT_system_prompt)
)

# Extract all full JSON blocks inside ```json ... ```
json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)

# Parse each block into dictionary
step_thoughts = [json.loads(block) for block in json_blocks]

print(step_thoughts)

# === RETRIEVAL PART ===

# Sleep with visible progress bar
def wait_with_progress(seconds):
    print(f"\n⏳ Waiting {seconds} seconds to avoid quota limits...\n")
    for _ in tqdm(range(seconds), desc="Sleeping", ncols=100):
        time.sleep(1)

final_answers = []

for step in step_thoughts:
    
    query = step["content"]
    
    if final_answers:
      previous_knowledge = "I know: " + " | ".join([ans["answer"] for ans in final_answers])
      query = f"{previous_knowledge}. Question: {step['content']}"
      
    print(f"\n🍳: {query}\n")
    print(f"🧠: {step}")
    
     # Retrieve relevant documents for this step
    docs = retriever.similarity_search(query) 
    
    # Combine all retrieved chunks
    context = "\n".join([doc.page_content for doc in docs])
    
    # Feed to Gemini for answering this step
    prompt = f"""
    Based on the following context, answer this question:
    
    Context:
    {context}
    
    Question:
    {query}
    """

    # 🕒 Wait 50 seconds before each request to avoid quota errors in free api 
    wait_with_progress(50)
    
    response = genai_client.models.generate_content(
    model="gemini-1.5-pro-latest",
    contents=prompt
    )

    step_answer = response.text.strip()
    final_answers.append({"question": query, "answer": step_answer})


# === GENERATION PART ===

# Prepare final answer from combined steps
combined_context = "\n".join([item["answer"] for item in final_answers])
final_user_query = step_thoughts[-1]["content"]  # last step is original query

final_prompt = f"""
Using the following reasoning and context, answer the final user query in a detailed yet simple way:

Context:
{combined_context}

Final Question:
{final_user_query}
"""

final_response = genai_client.models.generate_content(
    model="gemini-1.5-pro-latest",
    contents=final_prompt
)

# Display Final Answer
print("\n🎯 Final Answer to User Query:")
print(final_response.text.strip())





