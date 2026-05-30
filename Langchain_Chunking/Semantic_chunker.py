from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Choose embedding model (can be OpenAI, HuggingFace, etc.)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Create Semantic Chunker
semantic_chunker = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
)

# 3. Text for example
text = """
LangChain is a framework for building applications powered by language models.
It provides tools for Models, Prompts, Chains, Agents, and Memory.
LangChain also supports integrations with vector databases.
You can build retrieval-augmented generation (RAG) pipelines easily

My name is Raj kumar and i am learining lagnchain . hey this is dog . what are you doing gyes in space.
"""

# 4. Run semantic chunking
chunks = semantic_chunker.split_text(text)

# 5. Display results
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}\n")
