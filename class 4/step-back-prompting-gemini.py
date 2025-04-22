from pathlib import Path # File Path
from langchain_community.document_loaders import PyPDFLoader # Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text Splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Google Embedding
from langchain_qdrant import QdrantVectorStore # Vector Store

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

user_query = "What is FS Module in NodeJS?" # User Query

# === STEP-BACK PROMPT USING GEMINI ===

step_back_prompt = """
You are a helpful AI Assistant. 
Your task is to take the user's original query and convert it into a more conceptual or general question 
to provide better context for understanding and answering the original query.

Example: 
Query: Which year Mahatma Gandhi was born?
Step-Back: What is Mahatma Gandhi's personal history?

Query: Which skills are required to become a software developer?
Step-Back: How to become a software developer?
"""

# Call Gemini API to break down the user's query into sub-queries
step_back_response = genai_client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=f"Query: {user_query}",
    config=types.GenerateContentConfig(system_instruction=step_back_prompt)
)

conceptual_query = step_back_response.text.strip()
print("Step-Back (Conceptual) Query:", conceptual_query)

# Use conceptual_query instead of original user_query
relevant_chunks = retriever.similarity_search(query=conceptual_query)

# GENERATION PART 

final_system_prompt = f"""
You are a helpful assistant who answers the user's original query using the following conceptual context.
If you don't know the answer, just say you don't know — don't make up an answer.

Conceptual Context:
{[doc.page_content for doc in relevant_chunks]}
"""

final_response = genai_client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=user_query,
    config=types.GenerateContentConfig(system_instruction=final_system_prompt)
)

print("\nFinal Answer:\n")
print(final_response.text)
