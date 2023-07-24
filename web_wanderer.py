import os
import streamlit as st
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

@st.cache_resource
def public_settings(_stream_handler):

    """ LLM settings for public mode """

    # Vectorstore
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

    # LLM
    from langchain.chat_models import ChatOpenAI
    # TO DO: Stream formatting isn't great  
    llm = ChatOpenAI(temperature=0,streaming=True, callbacks=[stream_handler])

    return vectorstore, llm

@st.cache_resource
def private_settings(_stream_handler):

    """ LLM settings for privagte mode """

    # Vectorstore
    from langchain.vectorstores import Chroma
    from langchain.embeddings import GPT4AllEmbeddings
    vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(),persist_directory="./chroma_db_llama")

    # LLM
    from langchain.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="/Users/rlm/Desktop/Code/llama.cpp/llama-2-13b-chat.ggmlv3.q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,  # Context window
        max_tokens=1000,  # Max tokens to generate
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        # TO DO: Make this stream
        # callbacks=[stream_handler]? 
        callback_manager=callback_manager,
        verbose=True,
    )

    return vectorstore, llm

@st.cache_resource
def make_web_retriever(_vectorstore, _llm):

    """ Make web retriever """

    # Search 
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm, 
    search=search, 
    )

    return web_research_retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.sidebar.image("img/bot.png")

with st.sidebar.form("user_input"):

    # Pinecone params 
    mode = st.radio("`Operating mode:`",
                          ("Public",
                           "Private"),
                          index=0)
    
    submitted = st.form_submit_button("Set mode")

# Info 
st.header("`Web Wanderer`")
st.info("`I am a research assistant to answer questions by exploring, reading, and summarizing web pages."
    "I can be easily configured to use different moddes, public API or private (no data sharing).`")

# LLM
stream_handler = StreamHandler(st.empty())
if mode == "Public":
    vectorstore, llm = public_settings(stream_handler)
elif mode == "Private":
    vectorstore, llm = private_settings(stream_handler)
    
# Retriever
web_retriever = make_web_retriever(vectorstore, llm)

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_retriever)
    result = qa_chain({"question": question})
    st.info('Distilled answer:')
    st.info(result['answer'])
    st.info('Sources:')

        