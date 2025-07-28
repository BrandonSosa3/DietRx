import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_MASTER_CSV = "data/drug_food_interactions.csv"
FAISS_INDEX_DIR = "faiss_index"

def fetch_data():
    # Load the updated CSV with drug, food, and interaction columns
    df = pd.read_csv(DATA_MASTER_CSV)
    return df

def main():
    print("Loading data...")
    df = fetch_data()

    print(f"Total rows in CSV: {len(df)}")

    # Remove duplicates based on drug, food, and interaction to avoid redundancy
    df = df.drop_duplicates(subset=["drug", "food", "interaction"]).reset_index(drop=True)
    print(f"Rows after duplicate removal: {len(df)}")

    # Create documents combining drug, food, and interaction text for embedding
    documents = []
    for _, row in df.iterrows():
        doc_text = f"Drug: {row['drug']}\nFood: {row['food']}\nInteraction: {row['interaction']}"
        documents.append(Document(page_content=doc_text))
    print(f"Total documents before splitting: {len(documents)}")

    # Split documents into chunks for embedding (chunk size can be adjusted)
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(docs)}")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS vector store from docs and embeddings
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Remove old FAISS index folder if exists to avoid stale data
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"Removing old FAISS index at {FAISS_INDEX_DIR} ...")
        import shutil
        shutil.rmtree(FAISS_INDEX_DIR)

    # Save the new FAISS index locally
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"✅ Vector store created and saved as '{FAISS_INDEX_DIR}'")

if __name__ == "__main__":
    main()


