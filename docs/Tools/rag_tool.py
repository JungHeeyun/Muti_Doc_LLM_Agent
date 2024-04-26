from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def rag(faiss_index: int, query: str):
    chat = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)

    docs = faiss_index.similarity_search(query, k=3)
    
    results_content = ""
    for doc in docs:
        results_content += f"Page {doc.metadata['page']+1}: {doc.page_content}\n"
    
    messages = [
        SystemMessage(content=results_content.strip() + "\n Describe the content of each page based on user query keyword."),
        HumanMessage(content=query),
    ]

    return chat.invoke(messages)