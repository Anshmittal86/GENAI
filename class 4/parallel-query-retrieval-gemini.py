# Required imports
from pathlib import Path #Path
from langchain_community.document_loaders import PyPDFLoader #Langchain PDF Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Langchain Text Splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Langchain Google Embeddings
from langchain_qdrant import QdrantVectorStore #Langchain Qdrant Vector Store
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor #Multithreading
from itertools import chain #Flatten
import ast #Parsing

# === CONFIGURATION ===

# Initialize the Gemini client with your API key
genai_client = genai.Client(api_key='AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4')

# Setup the Google Embeddings using the required model and API key
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key="AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4"
)

# === INDEXING PART ===

# Set your file path (replace with the correct path to your PDF file)
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load the document from the PDF file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the document into smaller chunks for indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Create a new vector store and add the document chunks (if the collection doesn't already exist)
vector_store = QdrantVectorStore.from_documents(
    documents=[],  # Start with no documents in the store
    url="http://localhost:6333",  # URL to Qdrant vector store
    collection_name="GenAI",  # Name of your collection
    embedding=embeddings  # Embeddings for document search
)

# Add the split document chunks to the vector store
vector_store.add_documents(split_docs)

# === RETRIEVAL PART ===

# Setup the retriever using the existing collection from Qdrant
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="GenAI",  # Name of your collection in Qdrant
    embedding=embeddings
)

# Sample user query (can be dynamically set)
user_query = "What is fs module in nodejs?"

# === SUB-QUERY EXTRACTION USING GEMINI ===

# System prompt for breaking down the user's query into sub-queries
system_prompt_for_subqueries = """
You are a helpful AI Assistant. 
Your task is to take the user query and break it down into different sub-queries.

Rule:
Minimum Sub Query Length :- 3
Maximum Sub Query Length :- 5

Example:
Query: How to become GenAI Developer?
Output: [
    "How to become GenAI Developer?",
    "What is GenAI?",
    "What is Developer?",
    "What is GenAI Developer?",
    "Steps to become GenAI Developer."
]
"""

# Call Gemini API to break down the user's query into sub-queries
breakdown_response = genai_client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=f"Query: {user_query}",
    config=types.GenerateContentConfig(system_instruction=system_prompt_for_subqueries)
)

# Convert the Gemini response to a Python list (parse the output safely)
sub_queries = ast.literal_eval(breakdown_response.text.strip())
print("Sub Queries:", sub_queries)


# === PARALLEL VECTOR RETRIEVAL ===

# Function to retrieve relevant document chunks for each sub-query
def retrieve_chunks(query):
    return retriever.similarity_search(query=query)

# Use ThreadPoolExecutor to perform parallel retrieval of chunks for each sub-query
with ThreadPoolExecutor() as executor:
    all_chunks = list(executor.map(retrieve_chunks, sub_queries))

# Flatten the list of results (if there are multiple chunks per sub-query)
flattened_chunks = list(chain.from_iterable(all_chunks))

# Optionally remove duplicate chunks (based on content)
unique_chunks = list({doc.page_content: doc for doc in flattened_chunks}.values())

# === FINAL SYSTEM PROMPT FOR GEMINI ===

# Prepare the final system prompt with the unique relevant document chunks
final_system_prompt = f"""
You are a helpful assistant who answers the user's query using the following pieces of context.
If you don't know the answer, just say you don't know — don't make up an answer.

Context:
{[doc.page_content for doc in unique_chunks]}
"""

# Send the final request to Gemini for generating the response using the relevant context
final_response = genai_client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=user_query,  # The original user query
    config=types.GenerateContentConfig(system_instruction=final_system_prompt)
)

# Output the final response
print("\nFinal Answer:\n")
print(final_response.text)