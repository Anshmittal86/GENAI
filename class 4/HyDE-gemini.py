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

hypothetical_prompt = f"""
You are a helpful assistant who answer user's query.

Example:

Query: What is Node.js and how does it work?
Answer: Node.js is a JavaScript runtime environment that allows you to execute JavaScript code outside of a web browser. It is used for server-side development and building APIs.
"""

response = genai_client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=f"Query: {user_query}",
    config=types.GenerateContentConfig(system_instruction=hypothetical_prompt)
)

hypothetic_answer = response.text

relevant_chunks = retriever.similarity_search(
  query=hypothetic_answer,
)

# Generation part

final_system_prompt = f"""
You are a helpful assistant who answer user's query by using the following pieces of context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

context: {relevant_chunks}
"""

final_response = genai_client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=user_query,
    config=types.GenerateContentConfig(system_instruction=final_system_prompt)
)

print("\nFinal Answer:\n")
print(final_response.text)
