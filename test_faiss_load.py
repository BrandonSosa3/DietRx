from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def test_load():
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully!")

    query = "Check drug and food interactions for atorvastatin and grapefruit"
    results = vectorstore.similarity_search(query, k=3)
    print(f"Found {len(results)} results:")
    for doc in results:
        print(doc.page_content)

if __name__ == "__main__":
    test_load()
