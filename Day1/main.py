# Integrate our code with OpenAI API
import os
from constants import openai_key
from langchain_community.llms import OpenAI

import streamlit as st

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = openai_key

# Streamlit framework
st.title('Langchain demo with OpenAI API')
input_text = st.text_input("Search the topic that you want")

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.8)

# Process the input text
if input_text:
    response = llm(input_text)
    st.write(response)
