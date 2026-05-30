from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# defining the model
model= ChatOpenAI()

#defining the data
docs = {
    "doc1": "The Eiffel Tower was built in 1889 in Paris. It was designed by Gustave Eiffel...",
    "doc2": "The Taj Mahal was built in Agra by Shah Jahan in memory of Mumtaz Mahal..."
}

documents = [
    Document(page_content=docs["doc1"], metadata={"id": "doc1"}),
    Document(page_content=docs["doc2"], metadata={"id": "doc2"})
]

# splitting the texts 
child_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())

# Parent store that store full doc
docstore=InMemoryStore()

# now defining parentdocumentretrival

retriever=ParentDocumentRetriever(vectorstore=vector_store,docstore=docstore,child_splitter=child_splitter)

#Add documents (parent docs automatically chunked into children)
retriever.add_documents([
    Document(page_content=docs["doc1"], metadata={"id": "doc1"}),
    Document(page_content=docs["doc2"], metadata={"id": "doc2"})
])

results=retriever.invoke("when was the eiffel tower build?")
for r in results:
    print(r.page_content)