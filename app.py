import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile 

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama

# system_prompt = """
# Before You do Any action I want you to understand the file that is contained in Greek and translate it to English.
# You are an AI assistant tasked with providing detailed answers based solely on the given context.
# Your goal is to analyze the information provided and formulate a detailed answer back to the user.

# context will be passed as "Context:"
# user question will be passed as "Question:"

# To answer the question:
# 1. Thoroughly analyze the context, identifying key information relevant to the question.
# 2. Organize your thoughts and plan your response to ensure a logical flow of information.
# 3. Fomrmulate a detailed answer that directly address the question, using only the information provided in the context.
# 4. Ensure your answer is comprehensize, covering all relevant aspects found in the context.
# 5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

# Format your response as follows:
# 1. Use clear, concise language.
# 2. Organize your answer into paragraphs for readability.
# 3. Use bullet points or numbered lists where appropriate to break down complex information.
# 4. If relevant, include any headings or subheadings to structure your response.
# 5. Ensure proper grammar, punctuation, and spelling throughout your answer.

# After you finish all the accumptions and answers that you calculated translate the answers back to Greek!

# Important: Base your entire respone solely on the information provided in the context. Do not include any external
# knowledge or assumptions from external sources.
# """
system_prompt = """
 
 Είσαι Ενας ψηφιακός βοηθός που δουλειά σου ειναι να επιστρεφεις στον χρηστη , οτι σου ζηταει σχετικα
 με την γνωση που εχει ανεβασει στην βαση δεδομενων σου.

"""

def process_document(uploaded_file: UploadedFile)-> list[Document]:
    """
        Makes a temporary file as an object that we can
        "Welcome" our file that we want to upload into
        the system 
    """

    temp_file = tempfile.NamedTemporaryFile('w+b',suffix='.txt',delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    
    """
        It takes the temp_file loads it and then 
        throught the recursive text splitter it
        "chops" the words from our file(temp_file)
        into chunks!
    """
    try:

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n","\n",".","?","!"," ",""]
        )   
        return text_splitter.split_documents(docs)
    finally:
        
        os.unlink(temp_file.name)

EMBED_BATCH_SIZE = 32
OLLAMA_TIMEOUT_SECONDS = 600

def get_vector_collection() -> chromadb.Collection:

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )
    # Replace Chroma's default ollama client with one that won't read-timeout on
    # large batches or cold-start model loads.
    ollama_ef._client = ollama.Client(
        host="http://localhost:11434",
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw-space":"cosine"},
    )

def add_to_vector_collection(all_splits: list[Document],file_name:str):
    collection = get_vector_collection()
    documents , metadatas , ids = [],[],[]

    for idx,split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    total = len(documents)
    progress = st.progress(0.0, text=f"Embedding 0/{total} chunks...")
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        progress.progress(end / total, text=f"Embedding {end}/{total} chunks...")
    progress.empty()

    st.success(f"Added {total} chunks to the vector store!")

def query_collection(prompt: str,n_results: int =10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt],n_results=n_results)
    return results

def call_llm(context: str,prompt: str):
    response = ollama.chat(
        model="gemma4:e4b",
        stream=True,
        think=True,
        messages = [
            {
                "role":"system",
                "content":system_prompt,
            },
            {
                "role":"user",
                "content":f"Context: {context}, Question: {prompt} ",
            }


        ],
    )
    for chunk in response:
        msg = chunk["message"]
        if msg.thinking:
            yield ("think", msg.thinking)
        if msg.content:
            yield ("answer", msg.content)
        if chunk.get("done"):
            break

if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title = "Rag Question Answer")
        st.header("RAG QnA")
    
        uploaded_file = st.file_uploader(
             " Upload TXT,XLSX,PDF file for QnA ",
                type=["txt","xlsx","pdf"],
                accept_multiple_files = False
        )
        process = st.button("Process")

    if uploaded_file and process:
        # all_splits = process_document(uploaded_file)
        # st.write(all_splits)
        normalized_uploaded_file_name = uploaded_file.name.translate(
            str.maketrans({"-":"_",".":"_"," ":"_"})
        )
        all_splits = process_document(uploaded_file)
        add_to_vector_collection(all_splits, normalized_uploaded_file_name)
    
st.header("RAG QUESTION ANSWER")
prompt = st.text_area("Ask a question related to your document:")
ask = st.button("Ask")

if ask and prompt:
    results = query_collection(prompt)
    context = results.get("documents")[0]

    with st.expander("Thinking process", expanded=False):
        think_box = st.empty()
    answer_box = st.empty()

    think_buf, answer_buf = "", ""
    for kind, text in call_llm(context=context, prompt=prompt):
        if kind == "think":
            think_buf += text
            think_box.markdown(think_buf)
        else:
            answer_buf += text
            answer_box.markdown(answer_buf)
