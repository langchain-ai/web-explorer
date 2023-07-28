# Web Explorer

This is a lightweight app using the [Web Research Retriever](https://github.com/langchain-ai/langchain/pull/8102).

## Setup
You only need to supply a few things.

In `settings()` function, supply:

* Search: Select the search tool you want to use (e.g., GoogleSearchAPIWrapper). 
* Vectorstore: Select the vectorstore and embeddings you want to use (e.g., Chroma, OpenAIEmbeddings).
* Select the LLM you want to use (e.g., ChatOpenAI).

To use `st.secrets` set enviorment variables in `.streamlit/secrets.toml` file.
 
Or, simply add environemnt variables and remove `st.secrets`: 
```
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
os.environ["GOOGLE_CSE_ID"] = "YOUR_CSE_ID" 
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

```

For `GOOGLE_API_KEY` , you could get it from [this link](https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials).

For `GOOGLE_CSE_ID` , you could get it from [this link](https://programmablesearchengine.google.com/)

## Run
```
streamlit run web_explorer.py
```

Example output:
![example](https://github.com/langchain-ai/web-explorer/assets/122662504/f1383640-d089-492d-8757-ad743d34535f)