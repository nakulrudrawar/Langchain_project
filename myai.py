from secret_key import openai
import os
from langchain_community.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY']=openai

st.title("Tech Company info")
input_text = st.text_input("Enter Company Name")

#custom prompt template

first_input_prompt=PromptTemplate(
    input_variables=['CompanyName'],
    template = 'Tell me about {CompanyName}')

#Memory
Cname_memory=ConversationBufferMemory(input_key='CompanyName', memory_key='chat_history')
des_memory=ConversationBufferMemory(input_key='CompanyName', memory_key='chat_history')
ceo_memory=ConversationBufferMemory(input_key='CompanyName', memory_key='chat_history')

llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt, verbose=True,output_key='Description',memory=Cname_memory)

#second custom prompt template
second_input_prompt=PromptTemplate(
    input_variables=['CompanyName'],
    template = 'When was {CompanyName} established ')

chain2=LLMChain(llm=llm,prompt=second_input_prompt, verbose=True,output_key='est_date',memory=des_memory)

#third prompt template
third_input_prompt=PromptTemplate(
    input_variables=['CompanyName'],
    template = 'Who is ceo of {CompanyName} ')

chain3=LLMChain(llm=llm,prompt=third_input_prompt, verbose=True,output_key='ceo',memory=ceo_memory)

#Connecting the prompt template
parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['CompanyName'],output_variables=['Description','est_date','ceo'],verbose=True)

#giving output
if input_text:
    parent_chain({'CompanyName':input_text})

    with st.expander('Company Name'):
        st.info(Cname_memory.buffer)

    with st.expander('Establish Date'):
        st.info(des_memory.buffer)

    with st.expander('CEO Name'):
        st.info(ceo_memory.buffer)