from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(temperature=0.5)

parser= StrOutputParser()

prompt=PromptTemplate(
    template='Write a detaild report on {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="summarize the following {text}",
    input_variables=['text']
)

report=RunnableSequence(prompt,model,parser)

branch=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report,branch)

result=final_chain.invoke({"text":"russia vs ukrain"})

print(result)