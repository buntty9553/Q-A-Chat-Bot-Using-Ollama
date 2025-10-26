import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama



import os
from dotenv import load_dotenv
load_dotenv()


## 1.LangSmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot With Ollama"


## 2. Chat Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please response to the user's question as best as you can."),
        ("user", "Question: {question}")
    ]
)


def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


## 3. Streamlit APP

# Title
st.title("Enhanced Q&A Chatbot with Ollama")

## Select the model
engine = st.sidebar.selectbox("Select Ollama Model", ["tinyllama"])

## Set temperature and max tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main Interface
st.write("Go ahead and ask any question!")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get started.")