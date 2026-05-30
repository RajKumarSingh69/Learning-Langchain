from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv

# creating dummy data
docs = [
    Document(page_content="Python is a popular programming language."),
    Document(page_content="LangChain is a framework for building LLM apps."),
    Document(page_content="OpenAI provides GPT models for NLP tasks."),
]

# loading the environment variables
load_dotenv()

# defining the model
llm=ChatOpenAI()

# defining the embedding model
embd=OpenAIEmbeddings()

# defining the vector store
vector_store=Chroma.from_documents(docs,embd)

# here we are defining our retrieval with Similarity Threshold method
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)

# here we are defining our qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
query="What is python"
result=qa_chain.invoke(query)

print(result['result'])