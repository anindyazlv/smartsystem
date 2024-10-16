# Import streamlit for app development
import streamlit as st

# Import pandas for data manipulation
import pandas as pd
# Import transformer classes for generation
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import torch for datatype attributes
import torch
# Import the prompt wrapper for llama index
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms.huggingface import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings.langchain import LangchainEmbedding
# Bring in HF embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Change service context settings
from llama_index.core import Settings, set_global_service_context, VectorStoreIndex, download_loader
# Import dependencies to load documents
from pathlib import Path

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face
auth_token = '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'

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        cache_dir='./model/',
        token=auth_token, 
        torch_dtype=torch.float16,
        load_in_8bit=True
    )

    return tokenizer, model

tokenizer, model = get_tokenizer_model()

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of
the company.<</SYS>>
"""

# Query wrapper prompt
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)

# Create and download embeddings instance
embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = Settings(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# Set the service context
set_global_service_context(service_context)

# Add file upload functionality
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Load documents if a file is uploaded
if uploaded_file:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Download PDF Loader
    PyMuPDFReader = download_loader("PyMuPDFReader")
    # Create PDF Loader
    loader = PyMuPDFReader()
    # Load documents
    documents = loader.load(file_path=Path("temp.pdf"))

    # Convert PosixPath objects to strings in metadata
    for document in documents:
        document.metadata['file_path'] = str(document.metadata.get('file_path', ''))

    # Create an index from documents
    index = VectorStoreIndex.from_documents(documents)
    # Setup index query engine using LLM
    query_engine = index.as_query_engine()

    # Remove the temporary file
    Path("temp.pdf").unlink()

    # Create centered main title
    st.title('🦙 Llama 2 - RAG')
    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')

    # If the user hits enter
    if prompt:
        response = query_engine.query(prompt)
        # Extract and display the response text
        response_text = response.response
        st.write(response_text)

        # Display raw response object
        with st.expander('Response Object'):
            st.write(response)
        # Display source text
        with st.expander('Source Text'):
            st.write(response.get_formatted_sources())
