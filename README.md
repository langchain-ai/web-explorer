## Web Wanderer

This is a lightweight app using the [Web Research Retriever](https://github.com/langchain-ai/langchain/pull/8102).

You only need to supply a few thiings.

In `settings()` function, supply:

### Search
Select the search tool you want to use (e.g., GoogleSearchAPIWrapper).

### Vectorstore
Select the vectorstore you want to use (e.g., Chroma).

### LLM
Select the vectorstore you want to use (e.g., ChatOpenAI).

Then, run:

```
streamlit run web_explorer.py
```