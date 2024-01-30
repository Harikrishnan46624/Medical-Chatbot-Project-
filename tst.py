from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *




embeddings = download_hugging_face_embeddings()


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load the FAISS database with the embeddings
db = FAISS.load_local("vectorstore/db_faiss", embeddings=embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}


llm = CTransformers(model=r"E:\projects\Medical-Chatbot-Project-\model\llama-2-7b-chat.ggmlv3.q2_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



input = "what is acne"
result=qa({"query": input})
print("Response : ", result["result"])