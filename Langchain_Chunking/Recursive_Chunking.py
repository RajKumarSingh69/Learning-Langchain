from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
"""Instead of blindly cutting text every N chars, recursive splitting tries to keep chunks meaningful.

It splits hierarchically:

First by paragraphs

If too long → split by sentences

If still too long → split by words

Finally → by characters"""

prompt=PromptTemplate(
    template="Give me 10 line details information about the given topic.\{topic}",
    input_variables=["topic"]
)
load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

chain= prompt | model | parser

text=chain.invoke({"topic":"Global Warming"})

#now we are generate text from model and below we use Recursive Chunkning method

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n","\n",".", " ",""]
)

chunks=text_splitter.split_text(text)

print("Showin the chunks")

for i ,c in enumerate(chunks,1):
    print(f"{i}.{c}")