from utils.rag_chain import RAGCHAIN
from utils.embedding_store import Embeddings


vector_store = Embeddings()
retriever = vector_store.get_retriever(output_limit=3)
chain = RAGCHAIN(retriever=retriever)

print("print esc to close the console application")
print("to add documents to knowledge base, past the fie in data/KBI directory, and then type ingest_doc in input query")
while(True):
    print("USER : ", end=" ")
    input_query = input()
    cmd = input_query.lower()
    if(cmd == "esc"):
        break 
        
    elif(cmd == "ingest_doc"):
        print("Creating the Embeddings of the document")
        vector_store.ingest_docs()

    else:
        response = chain.chat(input_query) 
        print("\nBot : ", response)

    print('\n')