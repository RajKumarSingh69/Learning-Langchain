from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt=PromptTemplate(
    template="write a summary for the given poem \n {poem}",
    input_variables=['poem']
)
loader=TextLoader('cricket.txt',encoding='utf-8')

docs=loader.load()
chain=prompt | model | parser

result=chain.invoke({'poem':docs})

print(result)
