import os
import re
import json
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import qdrant_client

class ChaiDocsRAGSystem:
    def __init__(self, base_url="https://chaidocs.vercel.app"):
        self.base_url = base_url
        self.topics = {}
        self.documents = []
        self.vector_store = None
        self.llm = None
        self.collection_name = "chaidocs_collection"
        
    def scrape_content(self):
        """Scrape content from ChaiDocs YouTube track pages with improved error handling"""
        # Get the YouTube getting-started page
        start_url = f"{self.base_url}/youtube/getting-started"
        print(f"Accessing main page: {start_url}")
        
        main_page = requests.get(start_url)
        if main_page.status_code != 200:
            print(f"Failed to access the main page (status code: {main_page.status_code}). Please check the URL.")
            return False
            
        soup = BeautifulSoup(main_page.text, 'html.parser')
        
        # Find the sidebar navigation which contains all topic links
        # Try different possible class patterns for the navigation
        sidebar = None
        possible_nav_patterns = ['sidebar', 'nav', 'navigation', 'menu']
        
        for pattern in possible_nav_patterns:
            sidebar = soup.find('nav', class_=re.compile(pattern))
            if sidebar:
                print(f"Found navigation using pattern: {pattern}")
                break
                
        if not sidebar:
            print("Could not find navigation sidebar. Trying alternative approach...")
            # Try to find any nav element
            sidebar = soup.find('nav')
            
        if not sidebar:
            print("Still couldn't find navigation. Looking for any ul/ol elements that might contain links...")
            sidebar = soup.find(['ul', 'ol'])
            
        if not sidebar:
            print("Navigation extraction failed. Extracting from main page only.")
            # If we can't find proper navigation, just use the current page
            self._extract_content_from_page(start_url)
            return len(self.topics) > 0
        
        # Extract all links from the sidebar
        topic_links = []
        for link in sidebar.find_all('a'):
            href = link.get('href')
            if href and '/youtube/' in href:
                # Make sure we have absolute URLs
                if href.startswith('/'):
                    full_url = f"{self.base_url}{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = f"{self.base_url}/{href}"
                
                topic_links.append(full_url)
                
        # Remove duplicates while preserving order
        topic_links = list(dict.fromkeys(topic_links))
        print(f"Found {len(topic_links)} topic links.")
        
        # Process each topic link
        for link in topic_links:
            self._extract_content_from_page(link)
            # Add small delay to avoid overwhelming the server
            time.sleep(0.5)
            
        print(f"Successfully processed {len(self.topics)} topics.")
        return len(self.topics) > 0
    
    def _extract_content_from_page(self, url):
        """Extract content from a single page"""
        print(f"Processing: {url}")
        
        # Extract topic slug from URL for ID
        topic_slug = url.rstrip('/').split('/')[-1]
        
        # Fetch page content
        try:
            page = requests.get(url)
            if page.status_code != 200:
                print(f"Failed to access {url} (status code: {page.status_code})")
                return False
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return False
            
        # Parse the page
        page_soup = BeautifulSoup(page.text, 'html.parser')
        
        # Try multiple strategies to find the main content
        content_div = None
        
        # Strategy 1: Look for prose class (common in documentation sites)
        content_div = page_soup.find('div', class_=re.compile('prose'))
        
        # Strategy 2: Look for main content containers
        if not content_div:
            content_div = page_soup.find(['main', 'article'])
            
        # Strategy 3: Look for common content class names
        if not content_div:
            for class_pattern in ['content', 'main', 'article', 'post', 'body']:
                content_div = page_soup.find(['div', 'section'], class_=re.compile(class_pattern))
                if content_div:
                    break
        
        # Strategy 4: Fall back to body if needed
        if not content_div:
            content_div = page_soup.find('body')
        
        if not content_div:
            print(f"Could not find any content for {topic_slug}")
            return False
            
        # Get the title
        title_elem = page_soup.find('h1')
        if not title_elem:
            title_elem = page_soup.find('title')
        
        title = title_elem.text.strip() if title_elem else topic_slug
        
        # Extract text content focusing on elements that typically contain useful text
        content_elements = content_div.find_all(['p', 'li', 'h2', 'h3', 'h4', 'pre', 'code', 'blockquote'])
        
        if not content_elements:
            # If no elements found, use all text
            content = content_div.get_text(separator="\n")
        else:
            content = "\n".join([elem.get_text() for elem in content_elements])
        
        # Clean up content
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
        
        # Skip if we don't have meaningful content
        if len(content.strip()) < 50:
            print(f"Content too short for {topic_slug}, skipping")
            return False
        
        # Store the topic
        self.topics[topic_slug] = {
            'title': title,
            'content': content,
            'url': url
        }
        
        print(f"Successfully extracted: {title} ({len(content)} chars)")
        return True
    
    def prepare_documents(self):
        """Convert scraped content into LangChain documents"""
        if not self.topics:
            print("No topics found. Please run scrape_content() first.")
            return False
        
        # Text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        for topic_id, topic_data in self.topics.items():
            # Create metadata for this topic
            metadata = {
                "title": topic_data["title"],
                "topic_id": topic_id,
                "url": topic_data["url"]
            }
            
            # Split the content into chunks
            chunks = text_splitter.split_text(topic_data["content"])
            
            # Create documents for each chunk with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_id": i
                    }
                )
                self.documents.append(doc)
        
        print(f"Created {len(self.documents)} document chunks from {len(self.topics)} topics.")
        return True
    
    def build_vector_store(self, google_api_key=None):
        """Build a Qdrant vector store from the documents using Gemini embeddings"""
        if not self.documents:
            print("No documents prepared. Please run prepare_documents() first.")
            return False
            
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        # Check if API key is set
        if "GOOGLE_API_KEY" not in os.environ:
            print("ERROR: GOOGLE_API_KEY not found in environment variables.")
            print("Google API key is required for Gemini embeddings.")
            return False
            
        try:
            # Initialize Gemini embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            
            # Create vector store
            self.vector_store = Qdrant.from_documents(
                documents=self.documents,
                embedding=embeddings,
                collection_name=self.collection_name
            )
            
            print("Vector store created successfully with Qdrant and Gemini embeddings.")
            return True
            
        except Exception as e:
            print(f"Failed to build vector store: {str(e)}")
            return False
    
    def initialize_llm(self):
        """Initialize the LLM (Gemini) for answering questions"""
        # Check if API key is set (should already be set from build_vector_store)
        if "GOOGLE_API_KEY" not in os.environ:
            print("ERROR: GOOGLE_API_KEY not found in environment variables.")
            return False
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0.2,
            top_k=40,
            top_p=0.95,
            max_output_tokens=2048
        )
        
        # Create retrieval QA chain
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        # Define a custom prompt template
        template = """
        You are an assistant specialized in the ChaiDocs YouTube track documentation.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Always provide concise but comprehensive answers from the documentation.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("Gemini LLM initialized successfully.")
        return True
    
    def answer_question(self, query):
        """Answer a question using the RAG system"""
        if not self.vector_store:
            return "The vector store has not been initialized. Please build the vector store first."
        
        # First perform a similarity search to find relevant chunks
        try:
            relevant_chunks = self.vector_store.similarity_search(query, k=3)
            
            # If LLM not initialized, return the retrieved chunks directly
            if not self.llm:
                answer = f"Here's the most relevant information I found:\n\n"
                
                for i, doc in enumerate(relevant_chunks):
                    answer += f"--- From: {doc.metadata['title']} ---\n"
                    answer += doc.page_content
                    answer += f"\n\nSource: {doc.metadata['url']}\n\n"
                
                return answer
                
            # Use the QA chain to generate an answer using the query and relevant chunks
            result = self.qa_chain({"query": query})
            
            # Format the response
            answer = result["result"]
            
            # Add sources
            answer += "\n\nSources:"
            sources_added = set()  # To avoid duplicate sources
            
            for doc in result["source_documents"]:
                url = doc.metadata["url"]
                title = doc.metadata["title"]
                
                if url not in sources_added:
                    answer += f"\n- {title}: {url}"
                    sources_added.add(url)
            
            return answer
            
        except Exception as e:
            return f"Error processing your query: {str(e)}"
    
    def save_data(self, filename="chaidocs_rag_data.json"):
        """Save scraped data to a file"""
        with open(filename, 'w') as f:
            json.dump(self.topics, f)
        print(f"Data saved to {filename}")
        
    def load_data(self, filename="chaidocs_rag_data.json"):
        """Load scraped data from a file"""
        try:
            with open(filename, 'r') as f:
                self.topics = json.load(f)
            print(f"Loaded {len(self.topics)} topics from {filename}")
            return True
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return False
    
    def save_vector_store(self, directory="qdrant_data"):
        """Save the Qdrant vector store to disk"""
        if not self.vector_store:
            print("Vector store not created yet.")
            return False
        
        try:
            # Get the underlying Qdrant client
            client = self.vector_store._client
            
            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save the collection configuration
            collection_config = client.get_collection(self.collection_name)
            with open(f"{directory}/collection_config.json", "w") as f:
                json.dump(collection_config.dict(), f)
            
            # Save embeddings data
            embeddings_data = []
            for point in client.scroll(self.collection_name, limit=10000)[0]:
                embeddings_data.append({
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                })
            
            with open(f"{directory}/embeddings_data.json", "w") as f:
                json.dump(embeddings_data, f)
            
            print(f"Vector store saved to {directory}")
            return True
        except Exception as e:
            print(f"Failed to save vector store: {str(e)}")
            return False
    
    def load_vector_store(self, directory="qdrant_data", google_api_key=None):
        """Load the Qdrant vector store from disk"""
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        # Check if API key is set
        if "GOOGLE_API_KEY" not in os.environ:
            print("ERROR: GOOGLE_API_KEY not found in environment variables.")
            print("Google API key is required for Gemini embeddings.")
            return False
            
        try:
            # Check if the directory and files exist
            if not os.path.exists(directory) or \
               not os.path.exists(f"{directory}/collection_config.json") or \
               not os.path.exists(f"{directory}/embeddings_data.json"):
                print(f"Vector store data not found in {directory}")
                return False
            
            # Initialize Gemini embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            
            # Initialize Qdrant client (in-memory)
            client = qdrant_client.QdrantClient(":memory:")
            
            # Load collection configuration
            with open(f"{directory}/collection_config.json", "r") as f:
                collection_config = json.load(f)
                
            # Recreate collection
            client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=collection_config["config"]["params"]["vectors"],
            )
            
            # Load embeddings data
            with open(f"{directory}/embeddings_data.json", "r") as f:
                embeddings_data = json.load(f)
            
            # Batch upsert points
            points = []
            for point in embeddings_data:
                points.append(
                    qdrant_client.models.PointStruct(
                        id=point["id"],
                        payload=point["payload"],
                        vector=point["vector"]
                    )
                )
            
            # Upsert in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upsert(collection_name=self.collection_name, points=batch)
            
            # Create Qdrant vector store from existing client
            self.vector_store = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
            
            print(f"Vector store loaded from {directory} with {len(embeddings_data)} points")
            return True
        except Exception as e:
            print(f"Failed to load vector store: {str(e)}")
            return False
    
    def analyze_topics(self):
        """Print a summary of the collected topics"""
        if not self.topics:
            print("No topics available to analyze.")
            return
            
        print(f"\n--- Topic Analysis ({len(self.topics)} topics) ---")
        
        for idx, (topic_id, data) in enumerate(self.topics.items(), 1):
            title = data['title']
            content_len = len(data['content'])
            url = data['url']
            print(f"{idx}. {title} ({topic_id}) - {content_len} chars - {url}")

def main():
    print("ChaiDocs YouTube Track RAG System")
    print("================================")
    print("Using: Gemini Embeddings + Qdrant Vector Database")
    
    rag = ChaiDocsRAGSystem()
    
    # Get Google API key first as it's needed for both embeddings and LLM
    google_api_key = "AIzaSyB7auBxAy313T9TsXTnkkGArQ96W1anuH4"
    if not google_api_key:
        print("API key is required. Exiting.")
        return
    
    # Setup phase - First try to load existing data
    data_loaded = rag.load_data()
    vector_store_loaded = rag.load_vector_store("qdrant_data", google_api_key)
    
    # If either data or vector store is missing, rebuild from scratch
    if not data_loaded or not vector_store_loaded:
        print("\nBuilding the RAG system from scratch...")
        
        # Scrape content if needed
        if not data_loaded:
            print("Scraping content from ChaiDocs YouTube track...")
            if rag.scrape_content():
                rag.analyze_topics()
                rag.save_data()
            else:
                print("Failed to scrape content. Exiting.")
                return
        
        # Prepare documents and build vector store
        print("Preparing documents and building vector store with Gemini embeddings...")
        if rag.prepare_documents() and rag.build_vector_store(google_api_key):
            rag.save_vector_store()
        else:
            print("Failed to build vector store. Exiting.")
            return
    
    # Initialize LLM for question answering
    print("Initializing Gemini LLM for question answering...")
    rag.initialize_llm()
    
    print("\nRAG system ready! Ask questions about the ChaiDocs YouTube track.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nQuestion: ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            continue
            
        print("\nSearching for answer...")
        answer = rag.answer_question(query)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()