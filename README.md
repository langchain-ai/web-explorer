## Web Explorer

This is a lightweight app using the [Web Research Retriever](https://github.com/langchain-ai/langchain/pull/8102).

You only need to supply a few things.

In `settings()` function, supply:

* Search: Select the search tool you want to use (e.g., GoogleSearchAPIWrapper). 
* Vectorstore: Select the vectorstore and embeddings you want to use (e.g., Chroma, OpenAIEmbeddings).
* Select the LLM you want to use (e.g., ChatOpenAI).

To use `st.secrets` set enviorment variables in `.streamlit/secrets.toml` file.
 
Or, simply set all API keys and remove `st.secrets`: 
```
export GOOGLE_API_KEY=xxx
export GOOGLE_CSE_ID=xxx
export OPENAI_API_KEY=xxx
```

Run:
```
streamlit run web_explorer.py
```

Example output:
![example](https://github.com/langchain-ai/web-explorer/assets/122662504/f1383640-d089-492d-8757-ad743d34535f)