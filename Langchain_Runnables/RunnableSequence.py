from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(
    template="write a 5 line poeam about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']
)

model= ChatOpenAI()
parser=StrOutputParser()

# according to LCEL(Langchain Expression Language) we can write RunnableSequece as Pie( | )
#chain=prompt | model | parser
chain=RunnableSequence(prompt,model,parser,prompt2,model,parser)

result=chain.invoke({'topic':'cricket'})

print(result)