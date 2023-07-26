import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

@st.cache_resource
def settings(_stream_handler):

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore  
    embeddings_model = OpenAIEmbeddings()  
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[stream_handler])

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore_public,
    llm=llm, 
    search=search, 
    num_search_results=3
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

stream_handler = StreamHandler(st.empty())

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different moddes: public API or private (no data sharing).`")

# Set keys (use for deployed Streamlit app)
# For personal use, these can be hard-coded or set
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Make retriever and llm
web_retriever, llm = settings(stream_handler)

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
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    st.info('`Answer:`')
    st.info(result['answer'])
    st.info('`Source:`')
    st.info(result['sources'])
        