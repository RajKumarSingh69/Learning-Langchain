## Don't use this code for production level
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # used for loading the environmental variables

llm= OpenAI(model="gpt-3.5-turbo-instruct")

response=llm.invoke("Capital of bihar is")

print(response)