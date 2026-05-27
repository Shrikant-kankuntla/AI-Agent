import os
import traceback
import streamlit as st

from agent.graph import agent

# =====================================
# Environment settings
# =====================================

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

# =====================================
# Streamlit config
# =====================================

st.set_page_config(
    page_title="Coder Buddy",
    page_icon="🛠️",
    layout="wide",
)

# =====================================
# UI
# =====================================

st.title("🛠️ Coder Buddy")
st.caption("AI-powered project generator using LangGraph + Groq")

# =====================================
# Sidebar
# =====================================

with st.sidebar:
    st.header("⚙️ Settings")

    recursion_limit = st.slider(
        "Recursion Limit",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
    )

    st.markdown("---")

    st.info(
        "Example prompts:\n\n"
        "- Build a todo app using HTML CSS JS\n"
        "- Create a weather dashboard\n"
        "- Build a calculator app\n"
        "- Create a portfolio website"
    )

# =====================================
# Prompt input
# =====================================

prompt = st.text_area(
    "Enter your project idea",
    height=220,
    placeholder="Build a modern calculator using HTML, CSS, and JavaScript",
)

# =====================================
# Generate button
# =====================================

if st.button("🚀 Generate Project", use_container_width=True):

    if not prompt.strip():
        st.warning("Please enter a project prompt.")
        st.stop()

    try:

        with st.spinner("Generating project..."):

            result = agent.invoke(
                {"user_prompt": prompt},
                {"recursion_limit": recursion_limit},
            )

        st.success("✅ Project generated successfully!")

        # =====================================
        # Display Result
        # =====================================

        if isinstance(result, dict):

            # PLAN
            if "plan" in result:

                st.subheader("📋 Project Plan")

                try:
                    st.json(result["plan"].model_dump())
                except Exception:
                    st.write(result["plan"])

            # TASK PLAN
            if "task_plan" in result:

                st.subheader("🧠 Task Plan")

                try:
                    st.json(result["task_plan"].model_dump())
                except Exception:
                    st.write(result["task_plan"])

            # CODER STATE
            if "coder_state" in result:

                st.subheader("💻 Coding Progress")

                coder_state = result["coder_state"]

                try:
                    st.json(coder_state.model_dump())
                except Exception:
                    st.write(coder_state)

            # STATUS
            if "status" in result:

                st.subheader("📌 Status")
                st.code(str(result["status"]))

        else:
            st.write(result)

    except Exception as e:

        st.error("❌ Error while generating project")

        st.code(str(e))

        with st.expander("Full Traceback"):
            st.code(traceback.format_exc())
