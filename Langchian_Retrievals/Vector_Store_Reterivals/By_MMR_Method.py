from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# defining the custom docs
docs = [
    Document(page_content="Python is a popular programming language."),
    Document(page_content="LangChain is a framework for building LLM apps."),
    Document(page_content="OpenAI provides GPT models for NLP tasks."),
]

# defining the model
llm=ChatOpenAI()

#defining the embedding model
embd=OpenAIEmbeddings()

# defining the vector store
vector_store=Chroma.from_documents(docs,embd)

# definng the reterival with MMR Strategy
retriever= vector_store.as_retriever(search_type="mmr",
                                     search_kwargs={"k":2,"lambda_mult":0.7})

# defining the qa_chain
qa_chaIn=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query="Tell me about OpenAI"
result=qa_chaIn.invoke(query)

print(result)