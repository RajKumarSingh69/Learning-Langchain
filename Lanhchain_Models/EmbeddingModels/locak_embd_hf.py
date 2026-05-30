from langchain_huggingface import HuggingFaceEmbeddings

embd=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #

text="my name is raj kumar"
response=embd.embed_query(text)
print(str(response)) 