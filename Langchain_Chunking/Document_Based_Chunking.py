from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

loader=TextLoader("try.txt")
text=loader.load()

print("Our Original Text is\n")
print(text[0].page_content)

# now defining the document based chunking
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=5
)


#spliting the documents into chunks
docs=text_splitter.split_documents(text)

print("\nDocument Chunks with Metadata:")
for i, d in enumerate(docs[:5], 1):  # show first 5 chunks
    print(f"{i}. Content: {d.page_content}")
    print(f"   Metadata: {d.metadata}")