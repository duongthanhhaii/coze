import random
import string
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatCoze
import os
import json
from datetime import datetime
from streamlit_mic_recorder import speech_to_text
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
# from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import Image
import os
import torchtext
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
torchtext.disable_torchtext_deprecation_warning()

load_dotenv()

# app config
# st.set_page_config(page_title="Recommendation Chatbot", page_icon= "🤖")
st.set_page_config(page_title="Recommendation Chatbot", page_icon= "🤖")
st.title("Recommendation Chatbot")

def get_response(user_query, chat_history):
    template = """
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    coze = ChatCoze(
        # coze_api_base = os.getenv('URL'),
        coze_api_key = os.getenv('ACCESS_TOKEN'),
        bot_id = os.getenv('BOT_ID'),
        user="YOUR_USER_ID",
        conversation_id="YOUR_CONVERSATION_ID",
        streaming = False,
    )

    chain = prompt | coze
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })

def generate_random_id(length):
    characters = string.ascii_letters + string.digits 
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id


# session state
if "chat_history" not in st.session_state:
    st.session_state.id = generate_random_id(10)
    st.session_state.chat_history = [
        # AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
    
# display conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# text input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))

# speech input
speech_text = speech_to_text(
    language='en',
    start_prompt="🎤",
    stop_prompt="⏹",
    just_once=True,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)

if speech_text is not None and speech_text != "":
    st.session_state.chat_history.append(HumanMessage(content=speech_text))
    with st.chat_message("Human"):
        st.markdown(speech_text)
    with st.chat_message("AI"):
        response = get_response(speech_text, st.session_state.chat_history)
        st.markdown(response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))

# # image input
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# def image_captioning(img_file):
#     image = Image.open(img_file).convert("RGB")
#     inputs = processor(image, return_tensors="pt")
#     output = model.generate(**inputs)
#     return processor.decode(output[0], skip_special_tokens=True)

# img_file = st.file_uploader('Upload an image', type=['png', 'jpg'])
# if img_file is not None:
#     image = Image.open(img_file)
#     img_byte = img_file.getbuffer()
#     caption = image_captioning(img_file)
#     # print(caption)
#     query = 'Suggest me some products having this description: ' + caption

#     with st.chat_message("Human"):
#         st.image(image, caption = "Uploaded image") 
#     # print(img_file)
#     st.session_state.chat_history.append(HumanMessage(content=str(img_byte)))

#     with st.chat_message("AI"):
#         response = get_response(query, st.session_state.chat_history)
#         st.markdown(response.content)
#     st.session_state.chat_history.append(AIMessage(content=response))


# print(st.session_state.chat_history)


# export conversation
def export_conv_log(messages):
    time = datetime.now()
    conv_id = st.session_state.id
    conversation_file = f'conv_log/conv_{conv_id}.json'  

    messages_data = []
    for mess in messages:
        if isinstance(mess, AIMessage):
            role = 'Chatbot'
            content = mess.content
            messages_data.append({
                'role' : role,
                'content' : content
            })

        elif isinstance(mess, HumanMessage):
            role = 'User'
            content = mess.content
            messages_data.append({
                'role' : role,
                'content' : content
            })
    
    data = {
        'conversation_id' : conv_id,
        'time' : time.strftime("%Y-%m-%d %H:%M:%S"),
        'messages' : messages_data,
    }
    
    with open(conversation_file, 'w') as file:
        json.dump(data, file, indent=4)

export_conv_log(st.session_state.chat_history)