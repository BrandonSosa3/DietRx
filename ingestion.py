# ingestion.py

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load your CSV
df = pd.read_csv("data/drug_food_interactions.csv")

documents = []
for i, row in df.iterrows():
    text = (
        f"Drug: {row['Drug']}\n"
        f"Food: {row['Food']}\n"
        f"Interaction: {row['Interaction Description']}\n"
        f"Source: {row['Source URL']}"
    )
    documents.append(Document(page_content=text, metadata={"drug": row['Drug'], "food": row['Food']}))

# Use a lightweight HuggingFace model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vectorstore with FAISS
vectorstore = FAISS.from_documents(documents, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

print("✅ Vector store created and saved as 'faiss_index'")

