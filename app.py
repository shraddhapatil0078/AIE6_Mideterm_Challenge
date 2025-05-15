import os
import streamlit as st
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, List, Tuple, Union, TypedDict, Optional
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="RAG + Tavily Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📝 AI Agent with RAG + Tavily fallback")
st.markdown("""
This app combines RAG (Retrieval Augmented Generation) with Tavily search as a fallback mechanism.
Upload PDF documents and ask questions about their content.
""")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'tavily' not in st.session_state:
    st.session_state.tavily = None

# API Keys setup
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_key = st.text_input("OpenAI API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")

    if openai_key and tavily_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("✅ API keys configured!")
    else:
        st.warning("⚠️ Please enter both API keys to start")

# Main app logic
if openai_key and tavily_key:
    # File upload section
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Initialize Vector Store", type="primary"):
        with st.spinner("Processing documents..."):
            docs = []
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Load the PDF from the temporary file
                        loader = PyMuPDFLoader(tmp_file_path)
                        docs.extend(loader.load())
                        
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)
                        
                        st.success(f"✅ Processed {file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
            else:
                data_folder = 'data'
                if os.path.exists(data_folder):
                    pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pdf')]
                    for pdf_file in pdf_files:
                        try:
                            loader = PyMuPDFLoader(pdf_file)
                            docs.extend(loader.load())
                            st.success(f"✅ Processed {os.path.basename(pdf_file)}")
                        except Exception as e:
                            st.error(f"❌ Error processing {os.path.basename(pdf_file)}: {str(e)}")

            if docs:
                try:
                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                    split_docs = splitter.split_documents(docs)

                    # Initialize embeddings and vector store
                    embeddings = HuggingFaceEmbeddings(model_name="shradharp/legal-ft-658ea1b1-1d08-4417-8a4e-4920ae593642")
                    st.session_state.vector_store = Qdrant.from_documents(
                        split_docs,
                        embeddings,
                        location=":memory:",
                        collection_name="document_chunks"
                    )

                    # Initialize other components
                    st.session_state.llm = ChatOpenAI(model="gpt-4")
                    st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                    st.session_state.tavily = TavilySearchResults(max_results=5)

                    st.success(f"✅ Vector store initialized with {len(split_docs)} chunks!")
                except Exception as e:
                    st.error(f"❌ Error initializing vector store: {str(e)}")
            else:
                st.warning("⚠️ No documents found. Please upload PDF files or ensure the data folder contains PDFs.")

    # Question answering section
    if st.session_state.vector_store:
        st.header("❓ Ask Questions")
        query = st.text_input("Enter your question:", placeholder="What would you like to know?")
        
        if st.button("Ask Agent", type="primary") and query:
            with st.spinner("Thinking..."):
                try:
                    # Define the graph logic
                    retriever = st.session_state.retriever
                    llm = st.session_state.llm
                    tavily_search = st.session_state.tavily

                    RAG_PROMPT = """
                    You are an expert assistant that answers questions using ONLY the provided CONTEXT.
                    Do NOT make up any information.

                    CONTEXT:
                    {context}

                    USER QUESTION:
                    {question}

                    Instructions:
                    - If the context fully covers the answer, respond concisely and accurately.
                    - If the context is missing information needed to answer, respond exactly:
                      INSUFFICIENT_CONTEXT
                    """

                    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

                    rag_chain = (
                        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                        | RunnablePassthrough.assign(context=itemgetter("context"))
                        | {"response": rag_prompt | llm, "context": itemgetter("context")}
                    )

                    @tool
                    def rag_tool(input: str) -> dict:
                        """Tool to answer user query using document retrieval (RAG)."""
                        response = rag_chain.invoke({"question": input})
                        return {"answer": response["response"], "context": response["context"]}

                    @tool
                    def tavily_tool(input: str) -> dict:
                        """Tool to answer user query using Tavily web search."""
                        results = tavily_search.invoke({"query": input})
                        return {"results": results}

                    class AgentState(TypedDict):
                        input: str
                        rag_result: Optional[dict]
                        tavily_result: Optional[dict]

                    def should_fallback(state: AgentState) -> str:
                        rag_response = state.get("rag_result", {}).get("answer", "")
                        response_text = getattr(rag_response, "content", rag_response)
                        return "fallback" if response_text.strip().upper() == "INSUFFICIENT_CONTEXT" else "continue"

                    graph = StateGraph(AgentState)
                    graph.add_node("user_input", RunnableLambda(lambda state: {"input": state["input"]}))
                    graph.add_node("call_rag", RunnableLambda(lambda state: {"rag_result": rag_tool.invoke({"input": state["input"]})}))
                    graph.add_conditional_edges("call_rag", should_fallback, {"continue": END, "fallback": "call_tavily"})
                    graph.add_node("call_tavily", RunnableLambda(lambda state: {
                        "tavily_result": tavily_tool.invoke({"input": state["input"]}),
                        "rag_result": state["rag_result"]
                    }))
                    graph.set_entry_point("user_input")
                    graph.add_edge("user_input", "call_rag")
                    graph.add_edge("call_tavily", END)

                    agent = graph.compile()
                    result = agent.invoke({"input": query})

                    # Display results
                    st.subheader("📋 Agent Result")
                    rag_result = result.get("rag_result", {}).get("answer", "")
                    if hasattr(rag_result, "content"):
                        rag_result = rag_result.content

                    if rag_result and rag_result != "INSUFFICIENT_CONTEXT":
                        st.markdown(f"**RAG Answer:** {rag_result}")
                    else:
                        st.warning("RAG could not fully answer. Showing Tavily fallback:")
                        tavily_result = result.get("tavily_result", {}).get("results", [])
                        if tavily_result:
                            response_text = "\n\n".join([res.get("content", "") for res in tavily_result[:3] if res.get("content")])
                            st.markdown(f"**Tavily Answer:**\n\n{response_text}")
                        else:
                            st.info("No Tavily results found.")

                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")

else:
    st.info("Please enter both API keys in the sidebar to start using the app.")
