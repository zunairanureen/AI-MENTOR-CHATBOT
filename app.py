from itertools import zip_longest  # Used for combining lists of different lengths
import streamlit as st  # Streamlit library for creating web apps
from streamlit_chat import message  # Message component for displaying chat
from langchain.chat_models import ChatOpenAI  # Import OpenAI's chat model
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)  # Schema for defining system, human, and AI messages

# Set your OpenAI API key here
openai_key = "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"

# Set the Streamlit page configuration
st.set_page_config(
    page_title="SuperBOT"  # Title for the browser tab
)
st.title("AI Mentor")  # Title displayed on the web page

# Initialize session state variables if they don't already exist
if "generated" not in st.session_state:
    st.session_state["generated"] = []  # List to store AI responses

if "past" not in st.session_state:
    st.session_state["past"] = []  # List to store user queries

if "entered_prompt" not in st.session_state:
    st.session_state["entered_prompt"] = ""  # To temporarily hold the user's input

# Initialize the ChatOpenAI model with specified parameters
chat = ChatOpenAI(
    model="gpt-3.5-turbo",  # ChatGPT model to use
    temperature=0.5,  # Controls the randomness of the output
    max_tokens=None,  # No token limit here (default based on OpenAI model limits)
    timeout=None,  # No request timeout
    api_key=openai_key  # OpenAI API key for authorization
)

def build_message_list():
    """
    Build a list of messages that includes the initial system message,
    along with user and AI messages from previous conversations.
    """
    # Initial system message to set the chatbot's role and instructions
    zipped_messages = [SystemMessage(
        content="""Your name is AI mentor. You are an AI Technical Expert in Artificial Intelligence. 
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
    
    # Combine past user queries and AI responses, handling cases where lists have different lengths
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            # Add each user message to the message list as a HumanMessage
            zipped_messages.append(HumanMessage(content=human_msg))
        if ai_msg is not None:
            # Add each AI response to the message list as an AIMessage
            zipped_messages.append(AIMessage(content=ai_msg))
    
    return zipped_messages  # Return the compiled message list

def generate_response():
    """
    Generate an AI response using the ChatOpenAI model based on the compiled message list.
    """
    zipped_messages = build_message_list()  # Get the current conversation context
    ai_response = chat(zipped_messages)  # Generate AI response
    return ai_response.content  # Return the response content

def submit():
    """
    Capture user input when submitted and clear the input box.
    """
    st.session_state.entered_prompt = st.session_state.prompt_input  # Store the user input
    st.session_state.prompt_input = ""  # Clear the input field

# Input field for user to enter queries; calls submit() on change
st.text_input("You:", key="prompt_input", on_change=submit)

# If user has entered a prompt, process it
if st.session_state.entered_prompt != "":
    user_query = st.session_state.entered_prompt  # Assign user's input to a variable
    st.session_state.past.append(user_query)  # Save user query to session state 'past'

    # Generate AI response and save it to session state 'generated'
    output = generate_response()
    st.session_state.generated.append(output)

# Display the chat history in reverse order
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))  # Show AI's response
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Show user's query

