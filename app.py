import streamlit as st
from rag_chain import run_rag_query

st.set_page_config(page_title="DietRx", page_icon="💊")

st.title("💊 DietRx — AI Diet and Drug Interaction Advisor")
st.markdown("""
Enter your medications and foods to check for possible interactions.  
This tool uses AI to analyze medical information for you.
""")

# Input fields
drugs_input = st.text_area("Enter your medications (comma-separated):", value=st.session_state.get("drugs_input", ""))
foods_input = st.text_area("Enter your foods (comma-separated):", value=st.session_state.get("foods_input", ""))


with st.expander("💡 Need an example?"):
    st.markdown("""
    - **Medications:** atorvastatin, lisinopril  
    - **Foods:** grapefruit, bananas
    """)
    if st.button("Use Example"):
        st.session_state["drugs_input"] = "atorvastatin, lisinopril"
        st.session_state["foods_input"] = "grapefruit, bananas"


if st.button("Check Interactions"):
    if not drugs_input or not foods_input:
        st.error("Please enter both medications and foods.")
    else:
        drugs = [d.strip() for d in drugs_input.split(",")]
        foods = [f.strip() for f in foods_input.split(",")]

        st.write("Checking interactions for:")
        st.write(f"**Drugs:** {', '.join(drugs)}")
        st.write(f"**Foods:** {', '.join(foods)}")

        # Build user question string
        user_question = f"I am taking these medications: {', '.join(drugs)}. I often eat these foods: {', '.join(foods)}. Are there any interactions I should know about?"

        # Call your RAG pipeline
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


