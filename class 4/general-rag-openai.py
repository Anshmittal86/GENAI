from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Vector Store 
from langchain_qdrant import QdrantVectorStore

# OPEN AI
from openai import OpenAI
from openai import OpenAI as openai 

# Initialize OpenAI client
client = OpenAI()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load the document
loader = PyPDFLoader(file_path=pdf_path)

# Create a list of documents
docs = loader.load()

# print(docs[0])

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Create embeddings 
# Open AI Embeddings
embeddings = OpenAIEmbeddings(
  model="text-embedding-3-large",
  api_key = "OPENAI_API_KEY",
)

# Create a vector store - if collection exists
# vector_store = QdrantVectorStore.from_existing_collection(
#   url="http://localhost:6333",
#   collection_name="GenAI",
#   embedding=embeddings
# )

# create a vector store - if collection does not exist
vector_store = QdrantVectorStore.from_documents(
  documents=[],
  url="http://localhost:6333",
  collection_name="GenAI",
  embedding=embeddings
)

# Add the documents to the vector store
vector_store.add_documents(split_docs)

print("Injection part - END")

# Retrieval part
retriever = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="GenAI",
  embedding=embeddings
)

user_query = "What is Node.js and how does it work?"

# Search relevant chunks for the query
relevant_chunks = retriever.similarity_search(
  query=user_query,
)

# Generation part

system_prompt = f"""
You are a helpful assistant who answer user's query by using the following pieces of context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

context: {relevant_chunks}
"""

# Open AI
openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
    model="gpt-4",  # or "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
)

print(response['choices'][0]['message']['content'])