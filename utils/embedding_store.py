from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import shutil
from langchain_community.vectorstores import FAISS


class Embeddings:
    """
     This purpose of this class is to create embeddings of the documents and save them. 
    """
    def __init__(self, index_dir="faiss_index", index_path="index.faiss"):
        """
        index_path :- index path is the directory path where embeddings are saved
        """
        self.index_dir = index_dir 
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.index = None
        if os.path.exists(os.path.join(self.index_dir, index_path)):
            self.index = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)

    def ingest_docs(self, input_dir="data/KBI", output_dir="data/FinalKB"):
        input_docs = os.listdir(input_dir)
        if len(input_docs) == 0:
            print("No documents found in input directory.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)
        for file_name in input_docs:
            file_path = os.path.join(input_dir, file_name)
            loader = TextLoader(file_path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            # If index already loaded, add to it
            if self.index:
                self.index.add_documents(chunks)
            else:
                self.index = FAISS.from_documents(chunks, self.embeddings)
            shutil.move(file_path, os.path.join(output_dir, file_name))
            print(f"embeddings for file {file_name} are created and file is moved to {output_dir}")
        # Save index
        self.index.save_local(self.index_path)
        print(f"FAISS index updated and saved to '{self.index_path}'.")

    def get_retriever(self, output_limit=2):
        if(self.index):
            retriever = self.index.as_retriever(search_kwargs={"k":output_limit})
            return retriever 
        else:
            print("No index found")



if __name__ == "__main__":
    tmp = Embeddings()
    input_dir = r"data/KBI"
    output_dir = r"data/FinalKB"
    tmp.ingest_docs(input_dir, output_dir)