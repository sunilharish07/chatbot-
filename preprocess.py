import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.embeddings import HuggingFaceEmbeddings


# Set API key (use env in production)
os.environ["GOOGLE_API_KEY"] = "API"

CHROMA_DIR = "chroma_storage"
URLS = [
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology",
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Location",
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Academics",
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Departments",
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Admission_procedure",
    "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#National_level_FIDE_rated_chess_tournament"
]

def clean_wikipedia_content(url: str):
    """Fetch and clean content from Wikipedia page."""
    try:
        headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:

            print(f"⚠️ Failed to fetch {url} (Status: {response.status_code})")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "bodyContent"})

        if not content_div:
            print(f"⚠️ No content found at {url}")
            return None

        paragraphs = content_div.find_all("p")
        text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        if text:
            return Document(page_content=text, metadata={"source": url})
        else:
            print(f"⚠️ No valid text found at {url}")
            return None
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def scrape_and_embed():
    """Scrape Wikipedia pages, clean data, split into chunks, and embed in Chroma."""
    docs = []
    for url in URLS:
        print(f"🔗 Fetching: {url}")
        doc = clean_wikipedia_content(url)
        if doc:
            docs.append(doc)

    if not docs:
        print("❌ No documents fetched. Aborting embedding process.")
        return

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Filter empty documents after splitting
    split_docs = [doc for doc in split_docs if doc.page_content.strip()]
    if not split_docs:
        print("❌ No valid document chunks to embed.")
        return

    # Embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   

    # Quick check: test embedding on one doc
    test_vec = embeddings.embed_query("This is a test")
    print("✅ Embedding test vector length:", len(test_vec))

    # Save to Chroma
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("✅ Embeddings saved successfully in Chroma.")

    return vectorstore

if __name__ == "__main__":
    vectorstore = scrape_and_embed()

    if vectorstore:
        retriever = vectorstore.as_retriever()

        question = "What are the courses offered?"
        retrieved_docs = retriever.invoke(question)

        print(f"\n🔍 Retrieved {len(retrieved_docs)} docs for question: {question}\n")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Doc {i+1} ---\n{doc.page_content[:300]}...\n")
