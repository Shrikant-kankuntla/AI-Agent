import streamlit as st
import traceback

from agent.graph import agent

st.set_page_config(
    page_title="Coder Buddy",
    layout="wide"
)

st.title("🛠️ Coder Buddy")
st.write("AI-powered coding assistant")

prompt = st.text_area(
    "Enter your project idea",
    height=200,
    placeholder="Build a todo app using HTML CSS and JS"
)

recursion_limit = st.slider(
    "Recursion Limit",
    10,
    200,
    100
)

if st.button("Generate Project"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        try:
            with st.spinner("Generating..."):
                result = agent.invoke(
                    {"user_prompt": prompt},
                    {"recursion_limit": recursion_limit}
                )

            st.success("Project generated successfully!")

            st.json(result)

        except Exception as e:
            st.error(str(e))
            st.code(traceback.format_exc())