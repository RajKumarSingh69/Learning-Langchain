'''nstruction-based prompting is when you explicitly tell the model what to do by giving it a clear, direct instruction in natural language.
Instead of giving raw context or examples, you tell the model:

The task → What you want done.

The style → How you want it done.

The constraints → What rules it should follow.'''

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

#Instruction Based Prompting
prompt = ChatPromptTemplate.from_template("""
You are an expert Python developer.
Your task is to write Python code that sorts a list of integers in ascending order.
Requirements:
- Do not use the built-in sort() function.
- Explain your logic in simple terms after the code.
Return your answer in the following format:
Code:
<your code>
Explanation:
<your explanation>
""")

final_prompt=prompt.format() #format function is used for flling placeholders in the prompt template with actual values before sending it to the model.
response=model.invoke(final_prompt)
print(response.content)