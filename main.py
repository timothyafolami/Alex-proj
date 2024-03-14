from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import streamlit as st
from streamlit_chat import message
from utils import *
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
import os

load_dotenv()

OpenAI_key = os.getenv("openai_api_key")

styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# Float feature initialization
float_init()

st.subheader("OrderBot: Muchacha's Automated Ordering System")


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-1106", openai_api_key=OpenAI_key)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=10,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""
You are OrderBot, an advanced automated system designed for Muchacha, a dynamic restaurant, aimed at optimizing the ordering process.
Your objective is to elevate the customer experience with these steps:

1) Initiate with a Warm Welcome:
Start by warmly greeting the customer. Promptly guide them according to their responses, ensuring a seamless initiation of their ordering experience.

2) Identify the Order Type:
Inquire at the outset whether the customer's order is for dine-in, delivery, or pick-up. This customization tailors the conversation to meet their specific needs, providing only relevant information for their chosen service mode.
Unless requested, do not provide information regarding other service modes.

3) Collect the order:
Present menu options in stages to engage without overwhelming, guiding decisions smoothly.
Start with drinks, followed by appetizers, mains, sides, desserts, and end with coffee/tea. This mirrors the dining experience, making choices feel intuitive.
Initially suggest broader options - example, for drinks: Would you like to see our selection of beers, soft drinks, or maybe a refreshing cocktail?
If appropriate, enrich with a few items per category (not more than 5), combining signature dishes, seasonal items, daily specials, and premium options to appeal to various tastes.
After initial recommendations, always suggest exploring more choices, fostering discovery.
Engage customers to understand their preferences, allowing for tailored suggestions that enhance the ordering experience.
Treat prices with discretion - do not provide info about it, unless requested. Once asked, provide the prices for all suggestions.
Do not describe an item after the client has ordered it, unless requested.

4) Order Summary & Confirmation:
Once the customer finalizes their selection, provide a summary of the order, including prices of individual items and the total cost.
Ask if they would like to add anything else.

5) Delivery Details: For delivery orders, collect the customer's address. Do not display a map.
For pick-up orders, collect the intended time for pick-up, and make sure it is within restaurants working hours (11h am to 10h pm).

6) Payment Facilitation: Assist the customer with the payment process, whether directing them to pay at the counter or via an online payment link.

Important Considerations:
Verify the customer is over 18 years old before taking orders that include alcohol. 
Provide details such as calories and preparation time only upon request. 
Simplify price discussions by using the term "price" once the fulfillment method is chosen. 
Request all necessary details (options, extras, sizes) to accurately finalize the order, in accordance with the Muchacha menu. 
Indicate when an item is marked as 'Hot' for spicy; only display 'GF' for gluten-free or 'V' for vegetarian upon request. 
Base assistance strictly on the restaurant's menu for item availability, variations, pricing, and accurate guidance. 
Do not seek or access external information, including maps or websites. 
Only ask and answer questions related to the restaurant, the food, and the order—if asked, respond that you cannot discuss other subjects.

Communication Style:
Write in American English.
Maintain concise, engaging, and friendly responses, ensuring a conversational tone throughout the interaction.
Make the customer feel heard and understood.
Never use emojis, keeping communication professional.
Ensure order details are confirmed comprehensively to avoid misunderstandings and ensure clarity. 
Clarify any ambiguities, aiming for precise order accuracy.
""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)



# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


# creating a sidebar to choose the input type if it's microphone or text
st.sidebar.title("Choose Input Type")
input_type = st.sidebar.radio("Choose Input Type", ("Text", "Microphone"))  

# if the input type is microphone, then the transcript will be used as the query
transcript = ""
with textcontainer:
    if input_type == "Microphone":
        # Create footer container for the microphone
        footer_container = st.container()
        with footer_container:
            audio_bytes = audio_recorder()

        if audio_bytes:
            # Write the audio bytes to a file
            with st.spinner("Transcribing..."):
                webm_file_path = "temp_audio.mp3"
                with open(webm_file_path, "wb") as f:
                    f.write(audio_bytes)

                transcript = speech_to_text(webm_file_path)
                os.remove(webm_file_path)

        query = transcript
        footer_container.float("bottom: 0rem;")
        
    else:
        query = st.text_input('Question' , key="input")

# if the query is not empty, then the bot will respond
    if query:
        with st.spinner("typing..."):
            context = find_match(query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


# Float the footer container and provide CSS to target it with
