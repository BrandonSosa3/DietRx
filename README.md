DietRx: Drug & Food Interaction Advisor
DietRx is a local AI assistant that checks for potentially harmful interactions between medications and foods. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, natural-language explanations without requiring internet access.

Key Features:
100% offline – No API keys or external servers required
Privacy-first – All processing happens locally
Powered by TinyLlama (lightweight open-source LLM)
Built with RAG (FAISS + HuggingFace embeddings)
Simple Streamlit web interface

Technology:
Python – Core implementation
Streamlit – Web interface
FAISS – Vector similarity search
LangChain – RAG pipeline orchestration
HuggingFace – Sentence embeddings and TinyLlama model

Project Structure:
DietRx/
├── app.py                # Streamlit UI
├── rag_chain.py          # RAG logic
├── ingestion.py          # Data processing
├── data/                 
│   └── drug_food_interactions.csv  # Interaction database
├── faiss_index/          # Vector store
└── requirements.txt      # Dependencies


Installation:
1. Clone the repository:
git clone https://github.com/BrandonSosa3/DietRx.git
cd DietRx

2. Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt
Build the FAISS index:

4. Build the FAISS vector index:
python ingestion.py


5. Run the application:
streamlit run app.py


Example
Input:
Medications: warfarin, lisinopril
Foods: spinach, grapefruit

Output:
"Spinach is high in vitamin K, which can interfere with warfarin's effectiveness. Grapefruit may increase lisinopril blood levels, potentially causing side effects."
