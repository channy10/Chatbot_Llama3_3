import os
import json
import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Thiết lập giao diện Streamlit
st.set_page_config(page_title="Chat with Llama-3.3", page_icon="🤖", layout="centered")

# Đường dẫn API Groq
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Hàm tải API Key
def load_api_key():
    try:
        with open("config.json") as f:
            config_data = json.load(f)
        api_key = config_data.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("API Key không tồn tại. Vui lòng thêm vào config.json.")
        os.environ["GROQ_API_KEY"] = api_key
        return api_key
    except Exception as e:
        st.error(f"Lỗi khi tải API Key: {str(e)}")
        st.stop()

# Khởi tạo Vectorstore
@st.cache_resource
def setup_vectorstore():
    try:
        persist_directory = "vector_db_nlp"
        embeddings = HuggingFaceEmbeddings()
        st.info("Loading data and initializing Vectorstore...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo Vectorstore: {str(e)}")
        st.stop()

# Tạo pipeline truy vấn
def chat_chain(vectorstore):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        retriever = vectorstore.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            output_key="answer"  # Chỉ định khóa trả về
        )
        return chain
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo pipeline: {str(e)}")
        st.stop()

# Tải API Key và khởi tạo hệ thống
api_key = load_api_key()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = chat_chain(st.session_state.vectorstore)

# Giao diện Streamlit
st.title("📚 PDF Document QA System with LLaMA 3.3")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Lịch sử hội thoại trống ban đầu

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your question here!")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        # Gọi pipeline với chat_history và câu hỏi
        response = st.session_state.chat_chain.invoke({
            "question": user_input,
            "chat_history": [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "assistant"]
        })
        assistant_response = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý câu hỏi: {str(e)}")
