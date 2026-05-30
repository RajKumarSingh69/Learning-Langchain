from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(temperature=0.5)

parser= StrOutputParser()

def word_counter(text):
    return len(text.split())

prompt1=PromptTemplate(
    template="write 50 word joke on {topic}",
    input_variables=['topic']
)

joke_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_counter)
})

final_chain= joke_chain | parallel_chain

result=final_chain.invoke({'topic':"AI"})

print(result)
