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
    collection_name="GenAI",
    embedding=embeddings
)

# === USER QUERY ===
user_query = "What is FS Module in NodeJS?"

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


# === PARALLEL QUERY RETRIEVAL ===

# Function to retrieve relevant chunks for each sub-query
def retrieve_chunks(query):
    return retriever.similarity_search(query=query)

# Retrieve chunks in parallel
with ThreadPoolExecutor() as executor:
    all_chunks = list(executor.map(retrieve_chunks, sub_queries))

# Flatten and deduplicate chunks
flattened_chunks = list(chain.from_iterable(all_chunks))
unique_chunks = list({doc.page_content: doc for doc in flattened_chunks}.values())


# === GENERATION PART ===

final_system_prompt = f"""
You are a helpful assistant who answers the user's query using the following pieces of context.
If you don't know the answer, just say you don't know — don't make up an answer.

Context:
{[doc.page_content for doc in unique_chunks]}
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