from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser= JsonOutputParser()

template= PromptTemplate(
    template="Give me the name , age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    # this .get_format_instructions will additionally add the what format it required
    # it automatically filled the Give me the name , age and city of a fictional person 
    #  Return a JSON object ( this Json Line in our prompt)
)

# without using chains

#prompt= template.format()

#print(prompt)

#result=model.invoke(prompt)

#final_result=parser.parse(result.content)

# with using chains

chain= template | model | parser

#here chain.invoke always required a dictonary of inputs but in above code we can'specify any input_variable so that's why we pass here a blank dict 

final_result=chain.invoke({})

print(final_result)

print(type(final_result))