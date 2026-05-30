from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a tweet on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Create a Linkedin Post on {topic}",
    input_variables=['topic']
)

chain=RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkdin":RunnableSequence(prompt2,model,parser)
})

result=chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkdin'])
