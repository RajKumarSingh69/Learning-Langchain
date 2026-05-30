from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage,HumanMessage

load_dotenv()

model= ChatOpenAI()

chat_history=[
    SystemMessage(content="You are a helpfull Assistant")

]

'''chat_history=[] # we are creating a chat history so that model can rembember all the converstaion
# what if we want to recall something that we previosly told , it can answer

# below method is without any built-in langchain classes , and we are adding memory but it is not a good method 
while True:
    user_input=input("You")
    chat_history.append(user_input) # we are appendig the each user input in chat_history , so that user_input recording is here
    if user_input=='exit':
        #break
    result=model.invoke(chat_history) # we are also appending each model response whatever it is giving .
    chat_history.append(result.content)
    print("AI",result.content)
'''


#In this process we add that earlier we can't able to understand like which message is human messgae and which is AI generated message and which one is system message
# In this we call all the three types of messages in langchain 
# like HumanMessage,AIMessage, HumanMessage class
# by usin these and addin in out chat_history we can now see which one is by user and which one my AI
#Thi is the biggest flow 
while True:
    user_input=input("You")
    chat_history.append(HumanMessage(content=user_input)) # we are appendig the each user input in chat_history , so that user_input recording is here
    if user_input=='exit':
        break
    result=model.invoke(chat_history) # we are also appending each model response whatever it is giving .
    chat_history.append(AIMessage(content=result.content))
    print("AI",result.content)

print(chat_history)