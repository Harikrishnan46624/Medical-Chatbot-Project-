from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS


extracted_data = load_pdf(r"E:\projects\Medical-Chatbot-Project-\data")

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()



#Create Vector Database
def create_db(text_chunks, embeddings):
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)


create_db(text_chunks, embeddings)