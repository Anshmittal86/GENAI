from pathlib import Path #Path
from langchain_community.document_loaders import PyPDFLoader #Langchain PDF Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Langchain Text Splitter
from langchain_openai import OpenAIEmbeddings #Langchain OpenAI Embeddings
from langchain_qdrant import QdrantVectorStore #Langchain Qdrant Vector Store

from openai import OpenAI
from openai import OpenAI as openai 

from concurrent.futures import ThreadPoolExecutor #Multithreading
from itertools import chain #Flatten
import ast #Parsing


# === CONFIGURATION ===

openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize OpenAI client
client = OpenAI()

# Load OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="YOUR_OPENAI_API_KEY"
)

# === INDEXING ===

# Load your PDF
pdf_path = Path(__file__).parent / "your_pdf_file.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Initialize Qdrant vector store
vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="GenAI",
    embedding=embeddings
)

# Add documents to the vector store
vector_store.add_documents(split_docs)


# === RETRIEVAL PART ===
retriever = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="reciprocal_rank", # Name of your collection in Qdrant
  embedding=embeddings
)


user_query = "What is FS Module?" # User Query

# === BREAKDOWN TO SUB-QUERIES ===

sub_query_prompt = """
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

sub_query_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": sub_query_prompt},
        {"role": "user", "content": f"Query: {user_query}"}
    ]
)

# Convert the response text to a Python list
sub_queries = ast.literal_eval(sub_query_response['choices'][0]['message']['content'].strip())
print("Sub Queries:\n", sub_queries)

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

final_response = openai.ChatCompletion.create(
    model="gpt-4",  # or use "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_query}
    ]
)

# === OUTPUT ===

print("\nFinal Answer:\n")
print(final_response['choices'][0]['message']['content'])