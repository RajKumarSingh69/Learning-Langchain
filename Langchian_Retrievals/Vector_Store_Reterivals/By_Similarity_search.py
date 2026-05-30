from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Python is a popular programming language."),
    Document(page_content="LangChain is a framework for building LLM apps."),
    Document(page_content="OpenAI provides GPT models for NLP tasks."),
]

embed = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = Chroma.from_documents(docs, embed)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query = "What is LangChain used for?"
result = qa_chain.invoke(query)

print(result)
