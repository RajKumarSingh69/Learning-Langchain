from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

docs=[
    Document(page_content="Python is great for data science"),
    Document(page_content="Python is great for data science"),
    Document(page_content="Python is great for data science")]

#Embeddings
embed=OpenAIEmbeddings()
llm=ChatOpenAI()
#create chroma DB
faiss_db=FAISS.from_documents(docs,embed)
#
retriver=faiss_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriver,
    return_source_documents=True
)



query = "What is FAISS?"
response = qa_chain.invoke(query)

print(" Answer:", response["result"])
for doc in response["source_documents"]:
    print(" Source:", doc.page_content)