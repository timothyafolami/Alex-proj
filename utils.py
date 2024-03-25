from openai import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI as llm_openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

OpenAI_key = st.secrets.openai_api_key
# OpenAI_key = os.getenv('openai_api_key_1')

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

def load_docs():
    with open("id_items.txt", "r") as file:
        docs = file.read()
    return docs

def prepare_order(conversation,docs):
    conversation_history = conversation

    llm = llm_openai(api_key=OpenAI_key)
    docs = load_docs()
    prompt = PromptTemplate(
        input_variables=["conversation", 'docs'],
        template="""
        You are a helpful assistant that helps people order food online. 
        You have a conversation with a customer who wants to order food. 
        The conversation is as follows: {conversation}

        The menu information is as follows: {docs}
        Your task is to extract the order details from the conversation and place the order.
        Then you check the menu docs to get the items_ids of the order. 
        Your're returning the order details as a string like this "id_item" (column #) and "quantities" (column #).
        For example if the customer wants to order 2 pizzas and 1 burger, 
        you should something like this: item_id     quantity
                                            1           2
                                            2           1
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(conversation=conversation_history, docs=docs)
    return response


# print(load_docs())