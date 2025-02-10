import os
import streamlit as st
from groq import Groq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pypdf import PdfReader
import re
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time 


# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"  

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
print("Groq API key set.")

# Cached function to create a vectordb for the provided PDF files
@st.cache_data
def create_vectordb(files, filenames):
    with st.spinner("Creating vector database..."):
        print(f"Creating vectordb for files: {[f.name for f in files]}")
        vectordb = get_index_for_pdf([file.getvalue() for file in files], filenames)
    print("Vector database created.")
    return vectordb

# Cached function to create a vectordb for crawled content
@st.cache_data
def create_vectordb_from_crawled_data(urls):
    with st.spinner("Crawling URLs and creating vector database..."):
        print(f"Crawling URLs: {urls}")
        documents = asyncio.run(crawl_and_convert_to_docs(urls))
        print(f"Documents crawled: {len(documents)}")
        index = docs_to_index(documents)
    print("Vector database created from crawled data.")
    return index

# Parse PDF in parallel
def parse_pdf(file: BytesIO, filename: str):
    print(f"Parsing PDF file: {filename}")
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)
    print(f"Finished parsing PDF file: {filename}")
    return output, filename

def parse_pdfs_parallel(pdf_files, pdf_names):
    print("Starting parallel PDF parsing.")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(parse_pdf, pdf_files, pdf_names))
    print("Completed parallel PDF parsing.")
    return results

# Convert text to documents
def text_to_docs(text: list, filename: str):
    print(f"Converting text to documents for file: {filename}")
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
    print(f"Converted text to {len(doc_chunks)} documents.")
    return doc_chunks

# Create FAISS vector index
def docs_to_index(docs):
    print(f"Creating FAISS index for {len(docs)} documents.")
    # Assume you're still using Hugging Face embeddings for FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(docs, embeddings)
    print("FAISS index created.")
    return index

# Incrementally update FAISS index
def update_faiss_index(vectordb, new_docs):
    if vectordb is None:
        print("FAISS index is None. Creating new index.")
        vectordb = docs_to_index(new_docs)
    else:
        print(f"Updating existing FAISS index with {len(new_docs)} new documents.")
        vectordb.add_documents(new_docs)
    print("FAISS index updated.")
    return vectordb

# Get the index for PDF with parallel processing
def get_index_for_pdf(pdf_files, pdf_names):
    print("Getting index for PDFs.")
    results = parse_pdfs_parallel([BytesIO(pdf_file) for pdf_file in pdf_files], pdf_names)
    documents = []
    for text, filename in results:
        documents.extend(text_to_docs(text, filename))
    index = docs_to_index(documents)
    print("Index for PDFs created.")
    return index

# Utility function to get the base domain from a URL
def get_base_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

# Asynchronous function to fetch a single URL
async def fetch_url(session, url):
    async with session.get(url) as response:
        try:
            return await response.text()
        except UnicodeDecodeError:
            content = await response.read()
            return content.decode('ISO-8859-1')  # Fallback encoding

# Asynchronous function to fetch all links from a page
async def fetch_all_links(session, url):
    page_content = await fetch_url(session, url)
    soup = BeautifulSoup(page_content, 'html.parser')
    links = set()
    for link in soup.find_all('a', href=True):
        full_url = urljoin(url, link['href'])
        links.add(full_url)
    return links

# Asynchronous function to crawl URLs and convert to documents
async def crawl_and_convert_to_docs(urls):
    print(f"Crawling and converting documents from URLs: {urls}")
    documents = []
    visited_urls = set()
    base_domain = None

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_all_links(session, url) for url in urls]
        all_links = await asyncio.gather(*tasks)

        # Flatten list of sets and remove duplicates
        all_links = set(link for links in all_links for link in links)

        # Filter links to crawl
        if base_domain is None:
            base_domain = get_base_domain(urls[0])
            print(f"Base domain set to: {base_domain}")

        # Start crawling from the initial URLs
        crawl_tasks = []
        for link in all_links:
            if link not in visited_urls and link.startswith(base_domain):
                visited_urls.add(link)
                crawl_tasks.append(fetch_url(session, link))

        pages_content = await asyncio.gather(*crawl_tasks)

        for content, url in zip(pages_content, visited_urls):
            soup = BeautifulSoup(content, 'html.parser')
            crawled_text = soup.get_text()
            docs = text_to_docs([crawled_text], url)
            documents.extend(docs)
    
    print(f"Converted crawled data to {len(documents)} documents.")
    return documents

# Define the template for the chatbot prompt
prompt_template = """
You are BotX by Tushar, a dynamic virtual assistant designed to emulate human-like understanding and provide clear, concise answers. Your mission is to interpret user queries thoughtfully and respond accurately in a brief, helpful manner.

**Key Responsibilities:**
1. **Thoughtful Interpretation:** Carefully analyze each user query to understand the underlying intent.
  
2. **Concise and Clear Communication:** Respond with brief, straightforward answers that directly address the user's question. Avoid unnecessary elaboration and focus on delivering essential information in an easy-to-grasp format.

3. **Contextual Relevance:** Leverage the provided data and context for formulating responses. If a query lies outside your current scope, politely inform the user and suggest relevant resources.

4. **Consistent Accuracy:** Ensure that your responses are clear, reliable, and well-considered. Reflect on the user's query to deliver relevant, concise answers.

Your ultimate goal is to enhance the user experience by offering insightful, accurate, and easy-to-understand responses while maintaining brevity and relevance.

**Special Rule for Greetings:**
If the user query is a greeting (e.g., "hi," "hello," "hey"), respond sweetly with a short, friendly greeting without going in the provided context.

**Chat History:**
{chat_history}

**Context:**
{content}

**User Question:**
{user_query}

**BotX by Tushar Response:**
"""


st.set_page_config(page_title="Chatbot with Groq", page_icon="🤖")
st.title("Chat With Me 🤖")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    print("Initialized chat history.")

# Initialize vectordb if it doesn't exist in session state
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

# Upload PDF files
pdf_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

# Input for a single URL
url_input = st.text_input("Enter a website URL")

# Submit button for the URL input
url_submit = st.button("Submit URL")


# If PDFs are uploaded and vectordb is already created
if pdf_files and st.session_state["vectordb"]:
    st.write("PDFs are already stored. You can ask questions.")
    print("PDFs already stored.")
elif pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    print(f"Uploaded PDF files: {pdf_file_names}")
    vectordb = create_vectordb(pdf_files, pdf_file_names)
    st.session_state["vectordb"] = vectordb
    print("PDF vector database stored in session state.")
    st.write("PDFs uploaded successfully. Ask anything!")

# If URL is submitted and vectordb is already created
if url_submit and st.session_state["vectordb"]:
    st.write("URL is already stored. You can ask questions.")
    print("URL already stored.")
elif url_submit and url_input:
    urls = [url_input.strip()]  # Only one URL is submitted
    print(f"URL entered for crawling: {urls}")
    vectordb = create_vectordb_from_crawled_data(urls)
    st.session_state["vectordb"] = vectordb
    print("Crawled URL vector database stored in session state.")
    st.write("URL crawled successfully. Ask anything!")

# Display the chat history
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Get the user's question
user_query = st.chat_input("Ask a question about the PDF(s) or the URL")

if user_query:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("Please upload PDF(s) or enter a URL first.")
            print("No PDFs or URL provided yet.")
    else:
        with st.spinner("Generating response..."):
            # Fetch relevant documents based on user query
            start_time = time.time()
            docs = vectordb.similarity_search(user_query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            # Check if context is empty to avoid generating a response
            if not context.strip():
                with st.chat_message("assistant"):
                    st.write("I'm sorry, but I couldn't find relevant information to answer your question based on the provided content.")
            else:
                chat_history = "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in st.session_state["chat_history"]
                )
                # Use Groq to generate a response
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{
                        "role": "system",
                        "content": prompt_template.format(
                            content=context, user_query=user_query, chat_history=chat_history
                        ),
                    }],
                )
                
                elapsed_time = time.time() - start_time

                # Display the response and chat history
                with st.chat_message("user"):
                    st.write(user_query)
                with st.chat_message("assistant"):
                    st.write(response.choices[0].message.content)
                    st.write(f"Response time: {elapsed_time:.2f} seconds ⚡")

                # Update chat history
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_query}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response.choices[0].message.content}
                )
                print("Chat history updated.")                                                   