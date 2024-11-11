import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from itertools import zip_longest

# Set Streamlit page configuration
st.set_page_config(page_title="AI Mentor Chatbot")
st.title("AI Mentor - Chatbot")

# Initialize session state variables if they don't already exist
if "generated" not in st.session_state:
    st.session_state["generated"] = []  # List to store AI responses
if "past" not in st.session_state:
    st.session_state["past"] = []  # List to store user queries
if "entered_prompt" not in st.session_state:
    st.session_state["entered_prompt"] = ""  # To temporarily hold the user's input

# Load the GPT-2 model and tokenizer from Hugging Face
model_id = 'gpt2'  # GPT-2, a smaller and faster model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialize the text generation pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=200, max_new_tokens=50)

# System instructions
system_instructions = (
    "Your name is AI Mentor. You are an AI Technical Expert in Artificial Intelligence. "
    "Ask the user about their name before starting, and guide and assist students with their AI-related questions. "
    "1. Greet the user politely and ask how you can assist them with AI-related queries. "
    "2. Provide informative and relevant responses to questions about AI, ML, DL, NLP, CV, and related topics. "
    "3. Avoid sensitive content. Refrain from discrimination or harassment. "
    "4. Steer conversations back to AI topics as needed. "
    "5. Be patient and provide clear explanations. "
    "6. Respond politely if the user expresses gratitude or ends the conversation. "
    "7. Keep responses concise, with a maximum of 100 words. "
    "Reminder, your primary goal is to assist and educate students in the field of AI."
)

def build_prompt():
    """
    Build a prompt that combines system instructions with the conversation history.
    """
    # Include system instructions only once at the beginning
    prompt = system_instructions + "\n\nConversation:\n"
    
    # Combine past user queries and AI responses
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg:
            prompt += f"You: {human_msg}\n"
        if ai_msg:
            prompt += f"AI Mentor: {ai_msg}\n"
    
    # Add current user input if present
    if st.session_state.entered_prompt:
        prompt += f"You: {st.session_state.entered_prompt}\nAI Mentor:"
    
    return prompt

def generate_response():
    """
    Generate an AI response using the GPT-2 model based on the conversation history.
    """
    prompt = build_prompt()  # Get the conversation prompt
    ai_response = pipe(prompt, num_return_sequences=1)  # Generate response
    return ai_response[0]['generated_text'].split("AI Mentor:")[-1].strip()  # Extract response text

def submit():
    """
    Capture user input when submitted and clear the input box.
    """
    st.session_state.entered_prompt = st.session_state.prompt_input  # Store the user input
    st.session_state.prompt_input = ""  # Clear the input field

# Input field for user to enter queries; calls submit() on change
st.text_input("You:", key="prompt_input", on_change=submit)

# If user has entered a prompt, process it
if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt  # Assign user's input to a variable
    st.session_state.past.append(user_query)  # Save user query to session state 'past'

    # Generate AI response and save it to session state 'generated'
    output = generate_response()
    st.session_state.generated.append(output)

# Display the chat history in reverse order
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.markdown(f"**AI Mentor:** {st.session_state['generated'][i]}")
        st.markdown(f"**You:** {st.session_state['past'][i]}")
