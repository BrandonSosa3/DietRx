import streamlit as st
from rag_chain import run_rag_query

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

# Title and description
st.title("💊 DietRx — AI Diet and Drug Interaction Advisor")
st.markdown("""
Enter your medications and foods to check for possible interactions.  
This tool uses AI to analyze medical information for you.
""")

# Layout inputs side-by-side for better UX
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
        st.rerun()  # Refresh UI to show updated inputs

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


