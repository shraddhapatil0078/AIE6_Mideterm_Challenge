import os
import streamlit as st
from uuid import uuid4
import getpass
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

# Streamlit UI
st.set_page_config(page_title="RAG + Tavily Agent", layout="wide")
st.title("📝 AI Agent with RAG + Tavily fallback")

# Set API keys
openai_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
tavily_key = st.sidebar.text_input("Enter your Tavily API Key:", type="password")
# langsmith_key = st.sidebar.text_input("Enter your Langsmith API Key:", type="password")
# use finetuned_arctic_FT model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="finetuned_arctic_FT")
if openai_key and tavily_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_PROJECT"] = f"rag-agent-{uuid4().hex[:8]}"
    # os.environ["LANGCHAIN_API_KEY"] = langsmith_key

    # Upload PDFs or use data folder
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Initialize Vector Store"):
        docs = []
        if uploaded_files:
            for file in uploaded_files:
                loader = PyMuPDFLoader(file)
                docs.extend(loader.load())
        else:
            data_folder = 'data'
            pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pdf')]
            for pdf_file in pdf_files:
                loader = PyMuPDFLoader(pdf_file)
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        st.session_state.vector_store = Qdrant.from_documents(
            split_docs,
       #     OpenAIEmbeddings(model="text-embedding-3-small"),
            embeddings,
            location=":memory:",
            collection_name="usa_city_wiki_chunks"
        )

        st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")
        st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        st.session_state.tavily = TavilySearchResults(max_results=5)

        st.success(f"Vector store initialized with {len(split_docs)} chunks!")

    if "vector_store" in st.session_state:
        query = st.text_input("Enter your question:")
        if st.button("Ask Agent") and query:
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

else:
    st.info("Please enter all API keys to start.")
