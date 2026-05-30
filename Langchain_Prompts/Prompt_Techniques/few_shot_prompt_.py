from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

#few-short prompt
prompt = PromptTemplate(
    input_variables=["sentence"],
    template="""
Translate English sentences to Hindi.

Example 1:
English: How are you?
Hindi: आप कैसे हैं?

Example 2:
English: Good morning!
Hindi: शुभ प्रभात!

Now translate the following:
English: {sentence}
Hindi:
"""
)

final_prompt=prompt.format(sentence='Good Night ! Have Sweet Dreams')
response=model.invoke(final_prompt)
print(response.content)