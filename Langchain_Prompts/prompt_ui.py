from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st 
from dotenv import load_dotenv

load_dotenv()

st.header("Reaserch Tool")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
model=ChatOpenAI()

template = load_prompt('template.json') #loading our template

#fill the placehldrs
#prompt=template.invoke({
    #'paper_input':paper_input,
    #'style_input':style_input,
    #'length_input':length_input
#})

if st.button('Summarize'):
    #result=model.invoke(prompt)
    chain = template | model #using chain through which we are combing the template and model
    result=chain.invoke({ # we are invoking both template and model at one inseted of calling it two times by chain
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)