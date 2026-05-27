import os

from langchain_core.tools import tool

# =========================
# Project directory
# =========================

PROJECT_DIR = "generated_projects"

os.makedirs(PROJECT_DIR, exist_ok=True)

# =========================
# WRITE FILE
# =========================

@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    """

    full_path = os.path.join(PROJECT_DIR, path)

    os.makedirs(
        os.path.dirname(full_path),
        exist_ok=True
    )

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Successfully wrote file: {full_path}"

# =========================
# READ FILE
# =========================

@tool
def read_file(path: str) -> str:
    """
    Read a file.
    """

    full_path = os.path.join(PROJECT_DIR, path)

    if not os.path.exists(full_path):
        return ""

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# LIST FILES
# =========================

@tool
def list_files() -> list:
    """
    List all generated files.
    """

    file_list = []

    for root, dirs, files in os.walk(PROJECT_DIR):

        for file in files:

            file_list.append(
                os.path.join(root, file)
            )

    return file_list

# =========================
# CURRENT DIRECTORY
# =========================

@tool
def get_current_directory() -> str:
    """
    Get current working directory.
    """

    return os.getcwd()
