from openai import OpenAI
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

OpenAI_key = os.getenv("opena_api_key")

client = OpenAI(api_key=OpenAI_key)

embeddings = OpenAIEmbeddings(api_key=OpenAI_key)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def find_match(input):
    retriever = db.as_retriever()
    docs = retriever.invoke(input)
    return docs


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript