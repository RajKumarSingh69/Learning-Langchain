from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore         

from dotenv import load_dotenv

load_dotenv()
emb = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=emb)
docstore = InMemoryStore()

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",
)

doc_id = "1"

# --- Full Document ---
docstore.mset([
    (doc_id, Document(page_content="The Eiffel Tower was built in 1889 in Paris…"))
])

# --- Add 3 small vector representations ---
vectorstore.add_documents([
    Document(page_content="Eiffel Tower summary…", metadata={"doc_id": doc_id}),
    Document(page_content="Paris France monument", metadata={"doc_id": doc_id}),
    Document(page_content="1889 World's Fair tower", metadata={"doc_id": doc_id}),
])

# Query
results = retriever.invoke("When was the Eiffel Tower built?")
for r in results:
    print(r.page_content)

