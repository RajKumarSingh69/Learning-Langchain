from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#chat template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
load_dotenv()

#load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

#print(chat_history)

#crate a prompt
prompt=chat_template.invoke({'chat_history':chat_history,'query':'what is my order id'})

model=ChatOpenAI()

response=model.invoke(prompt)
print(response.content)
