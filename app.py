import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_chain import run_rag_query as original_run_rag_query

import streamlit as st
st.write("App started successfully!")


def load_known_items(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

known_drugs = load_known_items("data/known_drugs.txt")
known_foods = load_known_items("data/known_foods.txt")

# Page config
st.set_page_config(page_title="DietRx", page_icon="💊", layout="centered")

# Inject custom CSS styles
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
    /* Custom error box styling */
    .custom-error {
        background-color: #f44336;  /* Bright red */
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 1rem;
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

def show_custom_error_message(unknown_drugs, unknown_foods):
    message = "⚠️ Please check the spelling of the following entries:<br><br>"
    if unknown_drugs:
        message += f"<strong>Unknown medications:</strong> {', '.join(unknown_drugs)}<br>"
    if unknown_foods:
        message += f"<strong>Unknown foods:</strong> {', '.join(unknown_foods)}<br>"
    message += "<br>If spelling is correct, we are still working to add these to our database."

    st.markdown(f'<div class="custom-error">{message}</div>', unsafe_allow_html=True)

# Load embeddings and vectorstore once per session
if "vectorstore" not in st.session_state or "embeddings" not in st.session_state:
    with st.spinner("Loading AI models and index... This happens once per session."):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.load_local(
            "faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True
        )

# Wrapper to use cached vectorstore in session state
def run_rag_query(user_question: str):
    vectorstore = st.session_state.vectorstore
    relevant_docs = vectorstore.similarity_search(user_question, k=4)

    context = ""
    for doc in relevant_docs:
        context += f"{doc.page_content}\n\n"

    # This prompt is passed to your rag_chain.py’s model
    prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{context}

Question: {user_question}

Answer:"""

    # Call your existing function for generation
    return original_run_rag_query(user_question)

# UI Title and description
st.title("💊 DietRx — AI Diet and Drug Interaction Advisor")
st.markdown("""
Enter your medications and foods to check for possible interactions.  
This tool uses AI to analyze medical information for you.
""")

# Side-by-side input columns
col1, col2 = st.columns(2)

with col1:
    drugs_input = st.text_area(
        "Enter your medications (comma-separated):",
        value=st.session_state.get("drugs_input", ""),
        height=150,
    )

with col2:
    foods_input = st.text_area(
        "Enter your foods (comma-separated):",
        value=st.session_state.get("foods_input", ""),
        height=150,
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
        drugs = [d.strip().lower() for d in drugs_input.split(",")]
        foods = [f.strip().lower() for f in foods_input.split(",")]

        unknown_drugs = [d for d in drugs if d not in known_drugs]
        unknown_foods = [f for f in foods if f not in known_foods]

        if unknown_drugs or unknown_foods:
            show_custom_error_message(unknown_drugs, unknown_foods)
        else:
            st.write("Checking interactions for:")
            st.write(f"**Drugs:** {', '.join(drugs)}")
            st.write(f"**Foods:** {', '.join(foods)}")

            user_question = (
                f"I am taking these medications: {', '.join(drugs)}. "
                f"I often eat these foods: {', '.join(foods)}. "
                "Are there any interactions I should know about?"
            )

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
st.caption(
    "⚠️ This app is for informational purposes only. It does not provide medical advice. Always consult your doctor before making any health decisions."
)
