import openai

# Initialize the OpenAI GPT-3.5 model
openai.api_key = ''

def chatbot_response(conversation):
    print(conversation)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,

    )
    
    
    # Get the model's response
    chatbot_reply = response.choices[0].message['content']
    return chatbot_reply

# Initialize the conversation
conversation = [
    {"role": "system", "content": "You are a chatbot designed to assist children under the age of 8 with autism with their symptoms."},
]

while True:
    user_input = input("You: ")
    if "exit" in user_input.lower():
        break
    # Add the user's message to the conversation
    conversation.append({"role": "user", "content": user_input})
    
    # Get the chatbot's reply
    chatbot_reply = chatbot_response(conversation)
    conversation.append({"role": "assistant", "content": chatbot_reply})
    # Print the chatbot's response
    print("Chatbot:", chatbot_reply)

    # If the conversation should end, you can break out of the loop
    
