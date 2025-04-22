from pathlib import Path # File Path
from langchain_community.document_loaders import PyPDFLoader # Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text Splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Google Embedding
from langchain_qdrant import QdrantVectorStore # Vector Store

# GOGGLE GENERATIVE AI
from google import genai
from google.genai import types

from concurrent.futures import ThreadPoolExecutor #Multithreading
from itertools import chain #Flatten
import ast #Parsing


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

# Split the document into smaller chunks, Adjust chunk_size and chunk_overlap 
# according to your need
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Create a vector store - if collection exists
# vector_store = QdrantVectorStore.from_existing_collection(
#   url="http://localhost:6333",
#   collection_name="collection_name",
#   embedding=embeddings
# )

# Create a new vector store - if collection doesn't already exist
vector_store = QdrantVectorStore.from_documents(
  documents=[],
  url="http://localhost:6333",
  collection_name="reciprocal_rank", # Name of your collection in Qdrant
  embedding=embeddings
)

# Add the documents to the vector store
vector_store.add_documents(split_docs)

# === RETRIEVAL PART ===

retriever = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="reciprocal_rank", # Name of your collection in Qdrant
  embedding=embeddings
)

user_query = "What is FS Module?" # User Query

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

# === Reciprocal Rank Fusion ===

# Function to retrieve relevant document chunks for each sub-query
def retrieve_chunks(query):
    return retriever.similarity_search(query=query)

# Use ThreadPoolExecutor to perform parallel retrieval of chunks for each sub-query
with ThreadPoolExecutor() as executor:
    all_chunks = list(executor.map(retrieve_chunks, sub_queries))

# Helper to generate a unique ID for each chunk (or you can use doc.metadata['id'] if available)
def get_doc_id(doc):
    return doc.page_content.strip()[:50]  # Use first 50 characters as an ID

# Create rankings (lists of doc_ids per sub-query result)
rankings = []
for result in all_chunks:
    rankings.append([get_doc_id(doc) for doc in result])

# Reciprocal Rank Fusion
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]

# Get final ranked doc IDs
final_doc_ids = reciprocal_rank_fusion(rankings)

# Map doc IDs to actual chunks
doc_map = {get_doc_id(doc): doc for doc in chain.from_iterable(all_chunks)}
ranked_chunks = [doc_map[doc_id] for doc_id in final_doc_ids if doc_id in doc_map]

# === GENERATION PART ===

# Prepare the final system prompt with the top-ranked chunks
final_system_prompt = f"""
You are a helpful assistant who answers the user's query using the following pieces of context.
If you don't know the answer, just say you don't know — don't make up an answer.

Context:
{[doc.page_content for doc in ranked_chunks]}
"""

# Final call to Gemini using top-ranked documents
final_response = genai_client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=user_query,
    config=types.GenerateContentConfig(system_instruction=final_system_prompt)
)

# Output the final answer
print("\nFinal Answer:\n")
print(final_response.text)
