from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import os


os.environ["GOOGLE_API_KEY"] = "API"
# FastAPI app
app = FastAPI(title="Adithya College Chatbot API")

# Request & Response Models
class QuestionRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

class AnswerResponse(BaseModel):
    answer: str
    retrieved_context: List[str] = []  # include retrieved context for debugging

# Embeddings & Vector Store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_storage", embedding_function=embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for Adithya Institute of Technology. "
               "Use the following context to answer the question. "
               "Answer with formal replies for greetings. "
               "If the question is unrelated, respond with: "
               "\"I'm sorry, I can only answer questions about Adithya Institute of Technology.\""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Context:\n{context}")
])

doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

# API endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest):
    question = data.question.strip()
    if not question:
        return {"answer": "Please enter a valid question.", "retrieved_context": []}

    # Convert chat history
    chat_history = [
        HumanMessage(msg["content"]) if msg["role"] == "user" else AIMessage(msg["content"])
        for msg in data.chat_history
    ]

    # Retrieve documents from vector DB
    retrieved_docs = retriever.get_relevant_documents(question)
    context_texts = [doc.page_content for doc in retrieved_docs]

    # Optional: print/log for debugging
    print(f"Retrieved {len(retrieved_docs)} docs for question: {question}")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Doc {i+1} ---\n{doc.page_content[:300]}...\n")

    # Run RAG chain
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return {"answer": response["answer"], "retrieved_context": context_texts}
