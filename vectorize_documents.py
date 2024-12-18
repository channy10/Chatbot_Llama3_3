from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


# Hàm vector hóa tài liệu
def vectorize_documents():
    try:
        data_path = "Data1"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Thư mục '{data_path}' không tồn tại.")

        # Tải tài liệu từ thư mục
        loader = DirectoryLoader(path=data_path, glob="./*.pdf", loader_cls=UnstructuredFileLoader)
        documents = loader.load()

        if not documents:
            raise ValueError("Không tìm thấy tài liệu nào trong thư mục.")

        # Chia tài liệu thành các đoạn văn nhỏ
        embeddings = HuggingFaceEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)

        # Lưu trữ vector database
        persist_directory = "vector_db_nlp"
        vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings,
                                         persist_directory=persist_directory)
        print("Documents Vectorized")
        return vectordb
    except Exception as e:
        print(f"Lỗi khi xử lý tài liệu: {str(e)}")
        raise


if __name__ == "__main__":
    vectorize_documents()
