import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from cachetools import cached, LRUCache
import re

load_dotenv()
genai.configure(api_key="AIzaSyBrerOkXhkxcDIiSJa_rN98CODRlVq16M")
memory = ConversationBufferMemory(memory_key="chat_history")

# Cache for repeated queries
cache = LRUCache(maxsize=100)

def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Handle line breaks
    text = text.replace('\n', ' ')
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += preprocess_text(page.extract_text())
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    chroma_db = Chroma.from_texts(text_chunks,embeddings, persist_directory="./chroma_db")
    vector_store.save_local("faiss_index")

@cached(cache)
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and with your own knowledge, make sure to provide all the details, if the answer is not in
    provided context then try to give a the correct answer with your knowledge but make sure you don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        ch_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        faiss_retriever = new_db.as_retriever()
        chroma_retriever = ch_db.as_retriever()
        ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever,chroma_retriever], weights=[0.5, 0.5])
        docs = ensemble_retriever.invoke(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(response["output_text"])
        return response["output_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="Chat PDF- done by SHREE HARI B")
    st.header("Junior ML Engineer Assignment: Chat with PDF using Free LLM API and Streamlit", divider='rainbow')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Processing..."):
            response = user_input(user_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()