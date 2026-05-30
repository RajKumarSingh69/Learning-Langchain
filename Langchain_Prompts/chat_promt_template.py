from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

'''Chat_template=ChatPromptTemplate.from_messages([
    SystemMessage(content="you are a helpful{domain} expert"),
    HumanMessage(content="Explain in simple terms, what is {topic}")
])'''

#prompt=Chat_template.invoke({'domain':'cricket','topic':'Dusra'})
#print(prompt)
"""messages=[SystemMessage(content='you are a helpful {domain} expert', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='Explain in simple terms, what is {topic}', additional_kwargs={}, response_metadata={})]"

"""
#with this above method we can able to fill the placeholders as we are able to do in prompt template case
# now it will work by below method
Chat_template=ChatPromptTemplate([
    ('system',"you are a helpful{domain} expert"),
    ('human',"Explain in simple terms, what is {topic}")
])
prompt=Chat_template.invoke({'domain':'cricket','topic':'Dusra'})
print(prompt)

model=ChatOpenAI()
response=model.invoke(prompt)
print(response.content)