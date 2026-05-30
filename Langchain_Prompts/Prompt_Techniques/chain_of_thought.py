'''Chain-of-Thought (CoT) prompting is a technique where we explicitly ask the model to think step-by-step before giving the final answer'''

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

prompt=ChatPromptTemplate.from_messages([
    ("system","you are an expert mathematician and reasoning assistant"),
    ("user","question: {question} \n let's think step-by-setp")
])

final_prompt=prompt.format_messages(
    question="If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?"
)

response=model.invoke(final_prompt)
print(response.content)