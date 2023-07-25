import os
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

@st.cache_resource
def settings():

    # Vectorstore
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    vectorstore_public = Chroma(embedding_function=OpenAIEmbeddings())

    # LLM
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    os.environ["GOOGLE_CSE_ID"] = "xxx"
    os.environ["GOOGLE_API_KEY"] = "xxx"
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore_public,
    llm=llm, 
    search=search, 
    )

    return web_retriever, llm

st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different moddes: public API or private (no data sharing).`")

# Make retriever and llm
web_retriever, llm = settings()

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                           retriever=web_retriever)
    
    # Write answer and sources
    result = qa_chain({"question": question})
    st.info('`Answer:`')
    st.info(result['answer'])
    st.info('`Source:`')
    st.info(result['sources'])
        