import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use DistilGPT-2 for text generation
model_id = 'gpt2'  # A smaller and faster version
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialize the text generation pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=200, max_new_tokens=50)
chat = HuggingFacePipeline(pipeline=pipe)

# Set Streamlit page config
st.set_page_config(page_title='Conversational Q&A Chatbot')
st.header("Hey, Let's Chat")

# Initialize session state for chat flow if not already initialized
if "flowmessages" not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(
            content="""Your name is AI Mentor. You are an AI Technical Expert in Artificial Intelligence. 
            Ask the user about their name before starting, and guide and assist students with their AI-related questions. 
            1. Greet the user politely and ask how you can assist them with AI-related queries. 
            2. Provide informative and relevant responses to questions about AI, ML, DL, NLP, CV, and related topics. 
            3. Avoid sensitive content. Refrain from discrimination or harassment. 
            4. Steer conversations back to AI topics as needed. 
            5. Be patient and provide clear explanations. 
            6. Respond politely if the user expresses gratitude or ends the conversation. 
            7. Keep responses concise, with a maximum of 100 words.
            Reminder, your primary goal is to assist and educate students in the field of AI"""
        )]

# Function to generate response
def generate_response(question):
    # Add user question to the flow
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    
    # Convert the list of messages to a string that can be processed by the model
    chat_input = "\n".join([msg.content for msg in st.session_state['flowmessages']])
    
    # Generate the response using the HuggingFace model
    response = chat(chat_input)
    
    # Check if the response is a list and contains a dictionary with 'generated_text'
    if isinstance(response, list) and 'generated_text' in response[0]:
        generated_text = response[0]['generated_text']
    else:
        generated_text = response  # Assume response is a plain string if it's not in the expected format
    
    # Append the AI response to the session state
    st.session_state['flowmessages'].append(AIMessage(content=generated_text))
    
    return generated_text

# Text input widget for the user to enter a question
user_input = st.text_input("Ask a question about AI:", key="input")

# Check if the user presses the submit button
submit = st.button("Ask the question")

if submit and user_input:
    # Generate the response
    response = generate_response(user_input)
    
    # Clear the input field
    st.session_state["input"] = ""
    
    # Display the conversation history
    st.subheader("Conversation History:")
    for message in st.session_state['flowmessages']:
        if isinstance(message, HumanMessage):
            st.write(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"**AI Mentor:** {message.content}")
    
elif submit and not user_input:
    st.warning("Please enter a question before submitting.")
