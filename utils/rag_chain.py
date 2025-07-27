from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough 
import os 
from dotenv import load_dotenv 

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGCHAIN:
    """
        The purpose of this class to archestrate a rag chain 
    """
    def  __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(temperature=0.7, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
        self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("You are a helpful chatbot, who takes the context and question and give answer like a hhuman, talking to you feels so natural that, its hard to distinguish."),
                HumanMessagePromptTemplate.from_template("Use the following context:\n\n{context}\n\nAnswer the question:\n{query}, give answer in plain text. No Markdown ")
                ])
        self.get_content = RunnableLambda(lambda x : x.content)
        self.rag_chain = (
                RunnableMap({
                    "context": lambda q: "\n\n".join(doc.page_content for doc in self.retriever.invoke(q)),
                    "query": RunnablePassthrough()
                })
            | self.prompt
            | self.llm
            | self.get_content)


    def chat(self, query):
        response = self.rag_chain.invoke(query)
        return response


if __name__ == "__main__":
    from embedding_store import Embeddings
    emb_store = Embeddings()
    retriever = emb_store.get_retriever()
    rag = RAGCHAIN(retriever=retriever)
    rag.chat("summarize your knowledge base")