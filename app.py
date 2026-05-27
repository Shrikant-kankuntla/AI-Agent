import os
import zipfile
import shutil
import traceback
import streamlit as st

from agent.graph import agent

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="Coder Buddy",
    page_icon="🛠️",
    layout="wide",
)

PROJECT_DIR = "generated_projects"

os.makedirs(PROJECT_DIR, exist_ok=True)

# =====================================
# HELPERS
# =====================================

def clear_project_directory():
    """
    Remove old generated files.
    """

    if os.path.exists(PROJECT_DIR):
        shutil.rmtree(PROJECT_DIR)

    os.makedirs(PROJECT_DIR, exist_ok=True)


def create_zip(zip_path, folder_path):
    """
    Create ZIP file from generated project.
    """

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zipf:

        for root, dirs, files in os.walk(folder_path):

            for file in files:

                file_path = os.path.join(root, file)

                arcname = os.path.relpath(
                    file_path,
                    folder_path
                )

                zipf.write(file_path, arcname)


def get_generated_files():
    """
    Return all generated files.
    """

    generated_files = []

    for root, dirs, files in os.walk(PROJECT_DIR):

        for file in files:

            file_path = os.path.join(root, file)

            generated_files.append(file_path)

    return generated_files


# =====================================
# UI
# =====================================

st.title("🛠️ Coder Buddy")

st.caption(
    "AI-powered project generator using LangGraph + Groq"
)

st.info(
    """
💡 Example prompts:
- Build a modern todo app using React
- Create a weather dashboard
- Build a calculator using HTML CSS and JS
- Create a portfolio website
- Build a notes app with local storage
"""
)

# =====================================
# INPUT
# =====================================

prompt = st.text_area(
    "Enter your project prompt",
    height=220,
    placeholder="Build a weather dashboard using React",
)

recursion_limit = st.slider(
    "Recursion Limit",
    min_value=5,
    max_value=50,
    value=20,
)

# =====================================
# GENERATE BUTTON
# =====================================

if st.button(
    "🚀 Generate Project",
    use_container_width=True
):

    if not prompt.strip():

        st.warning("Please enter a prompt.")

        st.stop()

    try:

        # =====================================
        # Clear old project
        # =====================================

        clear_project_directory()

        # =====================================
        # Generate project
        # =====================================

        with st.spinner("Generating project..."):

            result = agent.invoke(
                {
                    "user_prompt": prompt
                },
                {
                    "recursion_limit": recursion_limit
                },
            )

        # =====================================
        # Success
        # =====================================

        st.success("✅ Project generated successfully!")

        # =====================================
        # Status
        # =====================================

        status = result.get("status", "DONE")

        st.subheader("📌 Status")

        st.code(status)

        # =====================================
        # Generated Files
        # =====================================

        st.subheader("📂 Generated Files")

        generated_files = get_generated_files()

        if not generated_files:

            st.warning("No files were generated.")

        else:

            for file_path in generated_files:

                relative_path = os.path.relpath(
                    file_path,
                    PROJECT_DIR
                )

                try:

                    with open(
                        file_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        content = f.read()

                except Exception:

                    content = "Unable to read file."

                with st.expander(
                    f"📄 {relative_path}",
                    expanded=False
                ):

                    st.code(
                        content,
                        language="javascript"
                    )

                    st.download_button(
                        label=f"⬇ Download {os.path.basename(file_path)}",
                        data=content,
                        file_name=os.path.basename(file_path),
                        mime="text/plain",
                        key=file_path,
                    )

        # =====================================
        # ZIP DOWNLOAD
        # =====================================

        zip_path = "generated_project.zip"

        create_zip(
            zip_path,
            PROJECT_DIR
        )

        with open(zip_path, "rb") as f:

            st.download_button(
                label="📦 Download Full Project ZIP",
                data=f,
                file_name="generated_project.zip",
                mime="application/zip",
            )

        # =====================================
        # Project Plan
        # =====================================

        if "plan" in result:

            st.subheader("📋 Project Plan")

            try:

                st.json(
                    result["plan"].model_dump()
                )

            except Exception:

                st.write(result["plan"])

    except Exception as e:

        st.error(
            "❌ Error while generating project"
        )

        st.code(str(e))

        with st.expander(
            "Full Traceback"
        ):

            st.code(
                traceback.format_exc()
            )
