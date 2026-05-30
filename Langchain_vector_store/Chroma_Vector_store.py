from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

load_dotenv()

docs=[
    Document(page_content="Python is great for data science"),
    Document(page_content="Python is great for data science"),
    Document(page_content="Python is great for data science")]

#Embeddings
embed=OpenAIEmbeddings()
llm=ChatOpenAI()
#create chroma DB
chroma_db=Chroma.from_documents(docs,embed)
#
retriver=chroma_db.as_retriever()
#qa chain
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriver,
    return_source_documents=True
)


# Query
query = "What is Chroma used for?"
response = qa_chain.invoke(query)

print("💡 Answer:", response["result"])
for doc in response["source_documents"]:
    print("📚 Source:", doc.page_content)
