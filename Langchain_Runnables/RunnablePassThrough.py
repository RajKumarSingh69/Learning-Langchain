from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a 5 line joke on {topic}\n",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Create a summapy of {topic}",
    input_variables=['topic']
)

res= prompt1 | model | parser

chain1=RunnableParallel({
    "joke":RunnablePassthrough(), 
    "explanation":RunnableSequence(prompt2,model,parser)
})

chain2=res | chain1
result=chain2.invoke({'topic':'AI'})

print(result)