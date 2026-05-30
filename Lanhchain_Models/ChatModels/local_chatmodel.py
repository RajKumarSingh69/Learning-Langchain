from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # ✅ Correct source
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "F:/huggingface_cache"

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create transformers pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.5,
    max_new_tokens=100
)

# Use HuggingFacePipeline from langchain_huggingface (✅ not from langchain_community)
llm = HuggingFacePipeline(pipeline=pipe)

# Now pass to ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Run the model
response = chat_model.invoke("What is the capital of Patna?")

# Print response
print(response.content)



