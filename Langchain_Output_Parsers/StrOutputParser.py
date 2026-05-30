from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Below process is without any parser and we can see that it little lengthy code
# and here we need to invoke the model and prompt multiple time.
# so making these easier we need to use the parsers.
'''
prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)'''

# below we can see that by using the parser and chains in langchain , it is very convinence code and easy to understand
parser= StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result= chain.invoke({'topic':'Black Hole'})
print(result)
