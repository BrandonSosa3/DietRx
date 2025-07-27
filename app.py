import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_chain import run_rag_query as original_run_rag_query

# Page config
st.set_page_config(page_title="DietRx", page_icon="💊", layout="centered")

# Inject custom CSS styles (your existing styles)
st.markdown(
    """
    <style>
    /* Background and main font */
    .main {
        background-color: #f7faf8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1a202c;
        padding: 1rem 2rem 2rem 2rem;
    }
    /* Title style */
    .css-1v3fvcr h1 {
        color: #2f855a;  /* medium green */
        font-weight: 700;
    }
    /* Text area style */
    textarea {
        border: 1.5px solid #38a169 !important;  /* green border */
        border-radius: 6px;
        font-size: 16px;
        padding: 0.5rem;
        background-color: #e6fffa;
    }
    /* Button style */
    div.stButton > button {
        background-color: #2f855a;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #276749;
        color: #f0fff4;
    }
    /* Info box styling */
    .stAlert {
        border-left: 6px solid #2f855a !important;
        background-color: #d1fae5 !important;
        color: #22543d !important;
        font-weight: 600;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2f855a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load embeddings and vectorstore once and cache them in session state
if "vectorstore" not in st.session_state or "embeddings" not in st.session_state:
    with st.spinner("Loading AI models and index... This happens once per session."):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.load_local("faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True)

# Redefine run_rag_query to use cached vectorstore & embeddings
def run_rag_query(user_question: str):
    vectorstore = st.session_state.vectorstore

    # Step 1: Find relevant documents using vector similarity
    relevant_docs = vectorstore.similarity_search(user_question, k=4)

    # Step 2: Format retrieved documents
    context = ""
    for doc in relevant_docs:
        context += f"{doc.page_content}\n\n"

    # Step 3: Create prompt
    prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{context}

Question: {user_question}

Answer:"""

    # Call your TinyLlama or HuggingFace model here (assuming rag_chain.run_rag_query did this)
    # But since you have your own rag_chain.py, you can also import and call it if it accepts vectorstore as param

    # For demonstration, just returning prompt for now
    # Replace with actual model generation call or import your existing run_rag_query and adjust it
    return original_run_rag_query(user_question)

# Title and description
st.title("💊 DietRx — AI Diet and Drug Interaction Advisor")
st.markdown("""
Enter your medications and foods to check for possible interactions.  
This tool uses AI to analyze medical information for you.
""")

# Layout inputs side-by-side
col1, col2 = st.columns(2)

with col1:
    drugs_input = st.text_area(
        "Enter your medications (comma-separated):", 
        value=st.session_state.get("drugs_input", ""),
        height=150
    )

with col2:
    foods_input = st.text_area(
        "Enter your foods (comma-separated):", 
        value=st.session_state.get("foods_input", ""),
        height=150
    )

with st.expander("💡 Need an example?"):
    st.markdown("""
    - **Medications:** atorvastatin, lisinopril  
    - **Foods:** grapefruit, bananas
    """)
    if st.button("Use Example"):
        st.session_state["drugs_input"] = "atorvastatin, lisinopril"
        st.session_state["foods_input"] = "grapefruit, bananas"
        st.rerun()

if st.button("Check Interactions"):
    if not drugs_input or not foods_input:
        st.error("Please enter both medications and foods.")
    else:
        drugs = [d.strip() for d in drugs_input.split(",")]
        foods = [f.strip() for f in foods_input.split(",")]

        st.write("Checking interactions for:")
        st.write(f"**Drugs:** {', '.join(drugs)}")
        st.write(f"**Foods:** {', '.join(foods)}")

        user_question = f"I am taking these medications: {', '.join(drugs)}. I often eat these foods: {', '.join(foods)}. Are there any interactions I should know about?"

        with st.spinner("Checking for interactions..."):
            try:
                response = run_rag_query(user_question)
                if not response.strip():
                    st.warning("⚠️ No answer generated. Try rephrasing your input or using different terms.")
                else:
                    st.success("✅ Interaction Analysis")
                    st.markdown("### 🧠 AI Recommendation")
                    st.markdown(f"```text\n{response}\n```")
            except Exception as e:
                st.error("Error while checking interactions.")
                st.error(str(e))

st.markdown("---")
st.caption("⚠️ This app is for informational purposes only. It does not provide medical advice. Always consult your doctor before making any health decisions.")
