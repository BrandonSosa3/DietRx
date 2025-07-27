import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_MASTER_CSV = "data/drug_food_interactions_master.csv"
FAISS_INDEX_DIR = "faiss_index"

def fetch_data():
    # For now, just load existing master CSV
    df = pd.read_csv(DATA_MASTER_CSV)
    return df

def main():
    print("Loading data...")
    df = fetch_data()

    print(f"Total rows in CSV: {len(df)}")

    # Remove duplicates based on key columns
    df = df.drop_duplicates(subset=["drug", "interaction"]).reset_index(drop=True)
    print(f"Rows after duplicate removal: {len(df)}")

    # Create documents for embedding
    documents = [Document(page_content=f"Drug: {row['drug']}\nInteraction: {row['interaction']}") for _, row in df.iterrows()]
    print(f"Total documents before splitting: {len(documents)}")

    # Increase chunk size to reduce total chunks
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(docs)}")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS vector store
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"Removing old FAISS index at {FAISS_INDEX_DIR} ...")
        import shutil
        shutil.rmtree(FAISS_INDEX_DIR)

    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"✅ Vector store created and saved as '{FAISS_INDEX_DIR}'")

if __name__ == "__main__":
    main()



