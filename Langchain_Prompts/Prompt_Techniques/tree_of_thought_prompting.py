from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.7)

tot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a reasoning assistant that explores multiple possible solutions before deciding."),
    ("human", """Question: {question}

Follow this process:
1. Think of at least 3 possible reasoning paths.
2. Write each path in detail.
3. Evaluate the pros and cons of each path.
4. Choose the best path and give the final answer.
""")
])

question = "A farmer has to transport a wolf, a goat, and a cabbage across the river. Only one item can be taken at a time. How can he do it without any of them being eaten?"

final_prompt = tot_prompt.format(question=question)
response = model.invoke(final_prompt)
print(response.content)
