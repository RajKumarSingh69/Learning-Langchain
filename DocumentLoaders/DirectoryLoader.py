from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader=DirectoryLoader(
    path="books",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.load()

print(len(docs))