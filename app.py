import streamlit as st

st.title("DietRx — AI Diet and Drug Interaction Advisor")

st.write("""
Enter the list of your medications and foods to check for potential interactions.
""")

# Input fields
drugs_input = st.text_area("Enter your medications (comma-separated):")
foods_input = st.text_area("Enter your foods (comma-separated):")

if st.button("Check Interactions"):
    if not drugs_input or not foods_input:
        st.error("Please enter both medications and foods.")
    else:
        drugs = [d.strip().lower() for d in drugs_input.split(",")]
        foods = [f.strip().lower() for f in foods_input.split(",")]

        st.write("Checking interactions for:")
        st.write(f"**Drugs:** {', '.join(drugs)}")
        st.write(f"**Foods:** {', '.join(foods)}")

        # TODO: Connect to RAG pipeline here to get interaction results
        st.info("Interaction checking logic coming soon!")
