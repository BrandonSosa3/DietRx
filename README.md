# DietRx: AI-Powered Drug & Food Interaction Advisor

DietRx is an intelligent AI assistant that checks for potentially harmful interactions between medications and foods. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, natural-language explanations with a modern Streamlit web interface.

## 🚀 Key Features

- **AI-Powered Analysis** - Uses advanced language models to provide detailed interaction explanations
- **Comprehensive Database** - 110+ drugs and 36+ foods with verified medical interactions
- **Smart Filtering** - Only shows relevant interactions for your specific drug-food combinations
- **Privacy-First** - All processing happens locally with no external API calls
- **Modern UI** - Clean, intuitive Streamlit interface with real-time feedback
- **Production Ready** - Deployed on Streamlit Cloud for easy access

## 🛠️ Technology Stack

- **Python 3.10+** - Core implementation
- **Streamlit** - Modern web interface
- **LangChain** - RAG pipeline orchestration
- **FAISS** - High-performance vector similarity search
- **HuggingFace** - Sentence embeddings and language models
- **Pandas** - Data processing and management

## 📁 Project Structure

```
DietRx/
├── app.py                    # Main Streamlit application
├── rag_chain.py             # RAG pipeline and AI logic
├── update_known_items.py    # Data management utility
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── data/
│   ├── drug_food_interactions.csv  # Medical interaction database
│   ├── known_drugs.txt            # Recognized medications
│   └── known_foods.txt            # Recognized foods
└── faiss_index/             # Vector search index
    ├── index.faiss
    └── index.pkl
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BrandonSosa3/DietRx.git
   cd DietRx
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment

The app is automatically deployed on Streamlit Cloud and available at: [dietrx.streamlit.app](https://dietrx.streamlit.app)

## 💡 Usage Example

**Input:**
- Medications: `metformin, atorvastatin`
- Foods: `grapefruit, alcohol`

**Output:**
```
✅ Interaction Analysis

🧠 AI Recommendation

Based on the medical information available, here are the key points about interactions between metformin, atorvastatin and grapefruit, alcohol:

1. Drug: atorvastatin
Food: grapefruit
Interaction: Grapefruit increases the blood levels of atorvastatin, which may increase the risk of side effects.

2. Drug: metformin
Food: alcohol
Interaction: Alcohol can increase the risk of lactic acidosis in patients taking metformin.

⚠️ Always consult your healthcare provider for personalized medical advice.
```

## 🔍 Supported Drugs & Foods

The system recognizes **110+ medications** and **36+ foods** including:

**Common Medications:** Metformin, Atorvastatin, Lisinopril, Warfarin, Digoxin, Levothyroxine, Simvastatin, and many more.

**Common Foods:** Alcohol, Grapefruit, Bananas, Apple, High-fiber foods, Calcium supplements, Cheese, Vitamin K, and many more.

## ⚡ Performance Notes

- **First load**: May take 1-2 minutes as AI models are loaded
- **Subsequent queries**: Fast responses due to intelligent caching
- **Memory optimized**: Uses lightweight models suitable for cloud deployment

## 🔒 Privacy & Security

- **100% Local Processing** - No data sent to external servers
- **No API Keys Required** - Self-contained application
- **Medical Disclaimer** - Always consult healthcare providers for medical decisions

## 📝 License

This project is open source and available under the MIT License.

---

**⚠️ Medical Disclaimer:** This application is for informational purposes only and does not provide medical advice. Always consult your healthcare provider before making any health decisions.
