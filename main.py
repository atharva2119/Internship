import streamlit as st
import os
from datetime import datetime
from typing import List
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    CombinedMemory
)
from langchain.memory.chat_memory import BaseChatMemory
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("AI Chatbot with Advanced Memory")

# Define the conversation prompt template
PROMPT_TEMPLATE = """You are a helpful and engaging AI assistant with access to different types of memory:
- Detailed recent conversation history
- Summary of earlier conversations
- Relevant context from documents

Use this information to provide natural, contextual responses. Maintain conversation continuity by referencing previous discussions when relevant.

Earlier Conversation Summary: {summary}
Recent Conversation History: {history}
Additional Context: {context}

Current Human Input: {input}
AI Assistant (respond naturally and conversationally):"""

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_start_time" not in st.session_state:
    st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = None
if "summary_memory" not in st.session_state:
    st.session_state.summary_memory = None

# Sidebar configuration
with st.sidebar:
    st.title("Settings")
    # API token input
    hf_api_token = st.text_input("Hugging Face API Token", type="password")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["google/flan-t5-large", "tiiuae/falcon-7b-instruct", "google/flan-t5-xl"],
        index=0
    )
    
    # Memory settings
    st.subheader("Memory Settings")
    max_token_limit = st.slider("Memory Token Limit", 100, 2000, 1000)
    use_summary = st.checkbox("Enable Conversation Summary", value=True)

    # Document upload for vector store
    uploaded_file = st.file_uploader("Upload Document for Knowledge Base", type=['txt'])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            with open("temp_doc.txt", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            loader = TextLoader("temp_doc.txt")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.vector_store = vector_store
            st.success("Document processed and vector store created!")
            
            os.remove("temp_doc.txt")

# Initialize or update LLM and conversation chain if API token is provided
if hf_api_token:
    try:
        # Initialize LLM
        llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=hf_api_token,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        )
        
        # Initialize different types of memory
        if not st.session_state.buffer_memory:
            # Recent conversation buffer
            st.session_state.buffer_memory = ConversationBufferMemory(
                memory_key="history",
                input_key="input",
                max_token_limit=max_token_limit,
                return_messages=True
            )
        
        if use_summary and not st.session_state.summary_memory:
            # Conversation summary memory
            st.session_state.summary_memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="summary",
                input_key="input"
            )
        
        # Combine memories
        memories: List[BaseChatMemory] = [st.session_state.buffer_memory]
        if use_summary and st.session_state.summary_memory:
            memories.append(st.session_state.summary_memory)
        
        combined_memory = CombinedMemory(memories=memories)
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["summary", "history", "context", "input"],
            template=PROMPT_TEMPLATE
        )
        
        # Create conversation chain
        st.session_state.conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=combined_memory,
            verbose=True
        )
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
else:
    st.warning("Please enter your Hugging Face API token to start.")
    st.session_state.conversation = None

# Chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
if prompt := st.chat_input("What would you like to discuss?"):
    if not hf_api_token:
        st.error("Please enter your Hugging Face API token in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Search vector store if available
    relevant_context = ""
    if st.session_state.vector_store is not None:
        results = st.session_state.vector_store.similarity_search(prompt, k=2)
        if results:
            relevant_context = "\n".join([doc.page_content for doc in results])
    
    try:
        # Generate response
        if st.session_state.conversation is None:
            st.error("Chatbot is not properly initialized. Please check your API token.")
            st.stop()
            
        with st.spinner("Thinking..."):
            # Get current memory states
            summary = st.session_state.summary_memory.load_memory_variables({})["summary"] if use_summary and st.session_state.summary_memory else ""
            
            response = st.session_state.conversation({
                "input": prompt,
                "context": relevant_context,
                "summary": summary
            })["text"]
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Memory visualization
with st.expander("View Memory Details"):
    st.subheader("Conversation Information")
    st.write(f"Started: {st.session_state.conversation_start_time}")
    
    # Display buffer memory
    st.subheader("Recent Conversation")
    if st.session_state.buffer_memory:
        messages = st.session_state.buffer_memory.chat_memory.messages
        for msg in messages:
            st.write(f"**{msg.type}**: {msg.content}")
    else:
        st.write("No recent conversation")
    
    # Display summary memory
    if use_summary and st.session_state.summary_memory:
        st.subheader("Conversation Summary")
        summary_data = st.session_state.summary_memory.load_memory_variables({})
        st.write(summary_data.get("summary", "No summary available"))

# Add memory management controls
st.sidebar.subheader("Memory Management")
if st.sidebar.button("Clear All Memory"):
    if st.session_state.buffer_memory:
        st.session_state.buffer_memory.clear()
    if st.session_state.summary_memory:
        st.session_state.summary_memory.clear()
    st.session_state.messages = []
    st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("Memory cleared!")
