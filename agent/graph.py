import re

from dotenv import load_dotenv

from langchain.globals import (
    set_verbose,
    set_debug,
)

from langchain_groq.chat_models import ChatGroq

from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.prompts import *
from agent.states import *

from agent.tools import (
    write_file,
    read_file,
)

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

# ==========================================
# DISABLE LOGS
# ==========================================

set_debug(False)
set_verbose(False)

# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ==========================================
# CLEAN MODEL OUTPUT
# ==========================================

def clean_code(content: str) -> str:

    if not content:
        return ""

    # remove markdown fences
    content = re.sub(r"```[a-zA-Z]*", "", content)
    content = content.replace("```", "")

    # remove fake filenames
    bad_lines = [
        "File:",
        "Filename:",
        "# File:",
        "// File:",
    ]

    cleaned = []

    for line in content.splitlines():

        stripped = line.strip()

        should_skip = any(
            stripped.startswith(x)
            for x in bad_lines
        )

        if not should_skip:
            cleaned.append(line)

    return "\n".join(cleaned).strip()

# ==========================================
# VALIDATION
# ==========================================

def validate_code(filepath: str, content: str):

    if not content.strip():
        return False

    if "```" in content:
        return False

    # ----------------------------
    # HTML VALIDATION
    # ----------------------------

    if filepath.endswith(".html"):

        required = [
            "<html",
            "</html>",
            "<body",
            "</body>",
        ]

        for item in required:

            if item.lower() not in content.lower():
                return False

    # ----------------------------
    # CSS VALIDATION
    # ----------------------------

    if filepath.endswith(".css"):

        if "{" not in content:
            return False

        if "}" not in content:
            return False

    # ----------------------------
    # JS VALIDATION
    # ----------------------------

    if filepath.endswith(".js"):

        banned_patterns = [
            ": string",
            ": number",
            ": void",
            "interface ",
            "type ",
            "document.getElementById('user-input')",
            "document.getElementById('calculate-button')",
            "TODO",
            "lorem ipsum",
        ]

        for pattern in banned_patterns:

            if pattern in content:
                return False

    return True

# ==========================================
# PLANNER
# ==========================================

def planner_agent(state: dict):

    user_prompt = state["user_prompt"]

    enhanced_prompt = f"""
{planner_prompt(user_prompt)}

IMPORTANT:

For simple apps like:
- calculator
- todo app
- portfolio
- weather app
- landing page

USE:
- html
- css
- javascript

DO NOT use:
- react
- typescript
- node
- express

unless explicitly requested.
"""

    response = llm.with_structured_output(
        Plan
    ).invoke(
        enhanced_prompt
    )

    return {
        "plan": response
    }

# ==========================================
# ARCHITECT
# ==========================================

def architect_agent(state: dict):

    plan = state["plan"]

    response = llm.with_structured_output(
        TaskPlan
    ).invoke(
        architect_prompt(
            plan=plan.model_dump_json()
        )
    )

    response.plan = plan

    return {
        "task_plan": response
    }

# ==========================================
# CODER
# ==========================================

def coder_agent(state: dict):

    coder_state = state.get(
        "coder_state"
    )

    # ----------------------------
    # INIT
    # ----------------------------

    if coder_state is None:

        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = (
        coder_state
        .task_plan
        .implementation_steps
    )

    # ----------------------------
    # DONE
    # ----------------------------

    if coder_state.current_step_idx >= len(steps):

        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    # ----------------------------
    # CURRENT TASK
    # ----------------------------

    current_task = steps[
        coder_state.current_step_idx
    ]

    # ----------------------------
    # EXISTING CONTENT
    # ----------------------------

    try:

        existing_content = read_file.invoke(
            {
                "path": current_task.filepath
            }
        )

    except Exception:

        existing_content = ""

    # ----------------------------
    # FILE TYPE RULES
    # ----------------------------

    extra_rules = ""

    if current_task.filepath.endswith(".js"):

        extra_rules = """
JAVASCRIPT RULES:
- Generate PURE JavaScript
- NEVER generate TypeScript
- NEVER use:
  : string
  : number
  : void
- NEVER use React syntax
- MUST work directly in browser
"""

    elif current_task.filepath.endswith(".html"):

        extra_rules = """
HTML RULES:
- Generate valid HTML5
- Link styles.css correctly
- Link script.js correctly
- Include visible UI
"""

    elif current_task.filepath.endswith(".css"):

        extra_rules = """
CSS RULES:
- Generate valid CSS
- Add responsive styling
"""

    # ----------------------------
    # PROMPT
    # ----------------------------

    prompt = f"""
You are an expert frontend engineer.

Generate code for ONLY ONE FILE.

PROJECT:
{coder_state.task_plan.plan.model_dump_json()}

TASK:
{current_task.task_description}

TARGET FILE:
{current_task.filepath}

EXISTING CONTENT:
{existing_content}

{extra_rules}

STRICT RULES:
- Output ONLY raw code
- No markdown
- No explanations
- No multiple files
- No fake filenames
- No TODO comments
- No placeholders
- Generate FULL code
- Ensure code is working
"""

    # ----------------------------
    # GENERATE
    # ----------------------------

    try:

        generated_code = ""

        max_retries = 3

        for _ in range(max_retries):

            response = llm.invoke(prompt)

            generated_code = clean_code(
                response.content
            )

            valid = validate_code(
                current_task.filepath,
                generated_code,
            )

            if valid:
                break

        if not generated_code.strip():

            generated_code = (
                f"/* Failed to generate "
                f"{current_task.filepath} */"
            )

        # ----------------------------
        # SAVE FILE
        # ----------------------------

        write_file.invoke(
            {
                "path": current_task.filepath,
                "content": generated_code,
            }
        )

    except Exception as e:

        return {
            "coder_state": coder_state,
            "status": f"ERROR: {str(e)}",
        }

    # ----------------------------
    # NEXT STEP
    # ----------------------------

    coder_state.current_step_idx += 1

    return {
        "coder_state": coder_state,
    }

# ==========================================
# BUILD GRAPH
# ==========================================

graph = StateGraph(dict)

graph.add_node(
    "planner",
    planner_agent
)

graph.add_node(
    "architect",
    architect_agent
)

graph.add_node(
    "coder",
    coder_agent
)

# ==========================================
# FLOW
# ==========================================

graph.add_edge(
    "planner",
    "architect"
)

graph.add_edge(
    "architect",
    "coder"
)

graph.add_conditional_edges(
    "coder",
    lambda s: (
        "END"
        if s.get("status") == "DONE"
        else "coder"
    ),
    {
        "END": END,
        "coder": "coder",
    },
)

# ==========================================
# ENTRY
# ==========================================

graph.set_entry_point(
    "planner"
)

# ==========================================
# COMPILE
# ==========================================

agent = graph.compile()

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "user_prompt":
            "Build a modern calculator using html css and javascript"
        },
        {
            "recursion_limit": 20
        }
    )

    print(result)
