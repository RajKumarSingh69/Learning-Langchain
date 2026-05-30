from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embd=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)
#response=embd.embed_query("My name Raj Kumar") # This embedd query is used for generating embeddings for single sentence
#print(str(response))

doc=[
    "This is Raj",
    "Patna",
    "Bihar"
]
res=embd.embed_documents(doc) #embd.embed_documents is used for generating embedding for full document
print(str(res))
print("\n")