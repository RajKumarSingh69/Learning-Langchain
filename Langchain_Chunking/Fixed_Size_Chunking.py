from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI()

prompt=PromptTemplate(
    template="Create 5 line summary on given topic /{topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain= prompt | model | parser

text=chain.invoke({'topic':"AI"})

# defining the fixed length chunking 
text_splitter=CharacterTextSplitter(
    separator="",
    chunk_size=40,
    chunk_overlap=10
)

chunks=text_splitter.split_text(text)

# here we are printing the splited chunks one by one
for i, c in enumerate(chunks,1):
    print(f"{i}.{c}")