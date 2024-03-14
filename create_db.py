from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv


load_dotenv()

OpenAI_key = os.getenv("opena_api_key")

embeddings = OpenAIEmbeddings(api_key=OpenAI_key)


def create_db(menu_list: str) -> FAISS:
    loader = TextLoader(menu_list)
    menu = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(menu)

    db = FAISS.from_documents(docs, embeddings)
    # save the database
    db.save_local("faiss_index")
    return db



if __name__ == "__main__":
    create_db('menu_items.txt')