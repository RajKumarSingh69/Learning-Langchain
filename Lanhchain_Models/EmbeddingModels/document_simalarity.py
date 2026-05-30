from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embed= OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)
doc = [
    "The quick brown fox jumps over the lazy dog.",
    "Data science is an exciting field.",
    "I love learning Python programming.",
    "Machine learning can solve real-world problems.",
    "The weather is sunny today.",
    "Artificial intelligence is the future.",
    "ChatGPT helps users with their queries.",
    "He is going to the market.",
    "Python is a powerful and versatile language.",
    "Deep learning is a subset of machine learning."
]
query="Tell me about Deep learning"

doc_embed=embed.embed_documents(doc) #here we got 10 vectors with 300d
query_embed=embed.embed_query(query) #here we got 1 veccotr with 300d

score=cosine_similarity([query_embed],doc_embed)[0]# it will give us top5 similarity scores which are very closest to our query
#now we fectch the similarity by convertng it into series with enumerate funcationa and the sort it based on index 1 so that we ca nget
# highest similarity score at last and we store that in variable
index,score= sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

# now we need to print our this index from doc
print(query)
print(doc[index])
print("similarity_score",score)
