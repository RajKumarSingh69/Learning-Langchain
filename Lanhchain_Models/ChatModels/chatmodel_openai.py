from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm= ChatOpenAI(model="gpt-4",temperature=1) #temp::- it control the creativity , lower the value actual and real the response

response=llm.invoke("who is PM of India")

#print(response)
"""content='As of my last update, the Prime Minister of India is Narendra Modi.
' additional_kwargs={'refusal': None} response_metadata={'token_usage':
 {'completion_tokens': 15, 'prompt_tokens': 12, 'total_tokens': 27, 
 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 
 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 
 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 
 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 
 'id': 'chatcmpl-C1wmuJPZkfMrDGjGdIfiKt2o8v1rb', 'service_tier': 'default', 
 'finish_reason': 'stop', 'logprobs': None} id='run--a9aa9d51-474e-44b2-8fcb-05ed5671a166-0' 
 usage_metadata={'input_tokens': 12, 'output_tokens': 15, 'total_tokens': 27, 
 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
"""
#Here we can see that after usin chat models and printing the direct result
# It is giving the mant things including our actual response in content
 # so we need to fetch the real output from content


print(response.content)