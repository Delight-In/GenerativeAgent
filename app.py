import os
import streamlit as st
from sentence_transformers import SentenceTransformer
# from files import *

# Assuming these are defined in your codebase
from files import (
    EMBED_MODEL,
    get_chroma_collection,
    index_document,
    load_llama_model,
    search_similar_chunks,
    build_prompt,
    run_llm,
)

st.set_page_config(page_title="📄 Document Q&A Chatbot", layout="wide")
st.title("📄 Local Document Q&A Chatbot")

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer(EMBED_MODEL)

if "llm" not in st.session_state:
    st.session_state.llm = load_llama_model()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

# Use default document if none uploaded
if uploaded_file:
    file_name = uploaded_file.name
    file_path = f"temp_uploads/{file_name}"
    os.makedirs("temp_uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
else:
    file_name = "data/Doc1.docx"
    file_path = os.path.join("default_docs", file_name)

    # You can include your default file in a `default_docs/` folder
    if not os.path.exists(file_path):
        st.warning("Default document not found. Please upload a file.")
        st.stop()
    else:
        st.info("📄 Using default document: `Doc1.docx`")

# Index the document only once
if "indexed_file" not in st.session_state or st.session_state.indexed_file != file_name:
    st.session_state.collection = get_chroma_collection(st.session_state.embedder)
    index_document(file_path, st.session_state.embedder, st.session_state.collection)
    st.session_state.indexed_file = file_name
    st.success(f"✅ Indexed: {file_name}")


    # --- Chat Interface ---
    st.subheader("Ask a question about the document")

    question = st.text_input("❓ Your question")

    if question:
        try:
            in_context, retrieved_chunks, _ = search_similar_chunks(
                question, st.session_state.embedder, st.session_state.collection
            )

            prompt = build_prompt(question, retrieved_chunks, in_context)
            answer = run_llm(st.session_state.llm, prompt)

            # Store in chat history
            st.session_state.chat_history.append((question, answer))

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# --- Display Chat History ---
if st.session_state.chat_history:
    st.subheader("🕘 Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**❓ You:** {q}")
        st.markdown(f"**🧠 Bot:** {a}")
        st.markdown("---")
