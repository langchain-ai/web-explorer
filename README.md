## Web Wanderer

This is a lightweight app for the LangChain [Web Research Retriever](https://github.com/langchain-ai/langchain/pull/8102).

You only need to supply a few thiings.

### Search
Supply search functionality e.g., Google - 
```
export GOOGLE_CSE_ID=xxx
export GOOGLE_API_KEY=xxx
search = GoogleSearchAPIWrapper() 
```

### Public API
Supply API key(s) e.g., OpenAI -
```
export OPENAI_API_KEY=sk-xxx
```

### Private
Follow [setup](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa) for local LLMs and supply path. 

### Run
streamlit run web_wanderer.py