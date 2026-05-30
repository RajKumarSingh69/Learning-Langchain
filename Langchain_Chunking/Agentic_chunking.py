from langchain_text_splitters import AgenticTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Use an LLM as the agent for deciding chunks
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the agentic splitter
splitter = AgenticTextSplitter.from_llm(
    llm,
    max_chunk_size=300,  # limit size
)

# Example text (long policy doc)
text = """
The company shall provide health insurance to all full-time employees.
Part-time employees may receive prorated benefits depending on hours worked.
Employees are entitled to 20 days of annual leave.
Unused leave can be carried forward up to 10 days.
Severe misconduct may lead to immediate termination.
"""

# Generate chunks
chunks = splitter.create_documents([text])

# Show chunks
for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
    print()
