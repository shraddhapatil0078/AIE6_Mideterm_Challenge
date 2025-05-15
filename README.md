# RAG + Tavily Agent

This is a Streamlit application that combines RAG (Retrieval Augmented Generation) with Tavily search as a fallback mechanism. The app allows users to upload PDF documents and ask questions about their content.

## Features

- PDF document upload and processing
- RAG-based question answering
- Tavily search fallback for questions not covered in the documents
- Secure API key management
- Modern Streamlit interface

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter your OpenAI and Tavily API keys in the sidebar
2. Upload PDF documents or use the default data folder
3. Click "Initialize Vector Store" to process the documents
4. Enter your questions in the text input
5. Click "Ask Agent" to get answers

## Requirements

- Python 3.10
- OpenAI API key
- Tavily API key

## Note

This app uses the HuggingFace embeddings model for document processing. Make sure you have sufficient memory and processing power for optimal performance.
