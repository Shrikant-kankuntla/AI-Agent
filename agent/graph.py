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

# =====================================
# LOAD ENVIRONMENT
# =====================================

load_dotenv()

# =====================================
# DISABLE LOGS
# =====================================

set_debug(False)
set_verbose(False)

# =====================================
# LLM
# =====================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# =====================================
# CLEAN MODEL OUTPUT
# =====================================

def clean_code(content: str) -> str:
    """
    Clean model output.
    """

    if not content:
        return ""

    lines = content.splitlines()

    cleaned_lines = []

    skip_prefixes = [
        "```",
        "---",
        "File:",
        "Filename:",
        "# File:",
        "// File:",
    ]

    for line in lines:

        stripped = line.strip()

        should_skip = any(
            stripped.startswith(prefix)
            for prefix in skip_prefixes
        )

        if not should_skip:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    return cleaned

# =====================================
# PLANNER AGENT
# =====================================

def planner_agent(state: dict) -> dict:
    """
    Convert user prompt into structured Plan.
    """

    user_prompt = state["user_prompt"]

    enhanced_prompt = f"""
{planner_prompt(user_prompt)}

IMPORTANT:
- If user asks for simple frontend apps,
  use:
  html, css, javascript

- Use React ONLY if explicitly requested

- For simple apps like:
  calculator
  todo app
  portfolio
  landing page
  weather app

  prefer plain:
  html + css + javascript
"""

    response = llm.with_structured_output(
        Plan
    ).invoke(
        enhanced_prompt
    )

    if response is None:

        raise ValueError(
            "Planner failed to generate plan."
        )

    return {
        "plan": response
    }

# =====================================
# ARCHITECT AGENT
# =====================================

def architect_agent(state: dict) -> dict:
    """
    Convert Plan into TaskPlan.
    """

    plan: Plan = state["plan"]

    response = llm.with_structured_output(
        TaskPlan
    ).invoke(
        architect_prompt(
            plan=plan.model_dump_json()
        )
    )

    if response is None:

        raise ValueError(
            "Architect failed to generate task plan."
        )

    response.plan = plan

    return {
        "task_plan": response
    }

# =====================================
# CODER AGENT
# =====================================

def coder_agent(state: dict) -> dict:
    """
    Generate code file-by-file.
    """

    coder_state: CoderState = state.get(
        "coder_state"
    )

    # =====================================
    # INITIALIZE STATE
    # =====================================

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

    # =====================================
    # STOP CONDITION
    # =====================================

    if (
        coder_state.current_step_idx
        >= len(steps)
    ):

        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    # =====================================
    # CURRENT TASK
    # =====================================

    current_task = steps[
        coder_state.current_step_idx
    ]

    # =====================================
    # READ EXISTING CONTENT
    # =====================================

    try:

        existing_content = read_file.invoke(
            {
                "path": current_task.filepath
            }
        )

    except Exception:

        existing_content = ""

    # =====================================
    # PROMPT
    # =====================================

    prompt = f"""
You are a senior software engineer.

You are generating ONLY ONE FILE.

PROJECT CONTEXT:
{coder_state.task_plan.plan.model_dump_json()}

CURRENT TASK:
{current_task.task_description}

TARGET FILE:
{current_task.filepath}

EXISTING FILE CONTENT:
{existing_content}

STRICT RULES:
- Generate code ONLY for:
  {current_task.filepath}

- NEVER generate multiple files
- NEVER include markdown
- NEVER include ``` fences
- NEVER explain anything
- Return ONLY raw code

IMPORTANT PROJECT RULES:

- Every HTML project MUST contain:
  - index.html
  - style.css
  - script.js

- index.html MUST:
  - contain valid HTML5
  - properly link CSS and JS
  - contain visible UI

- CSS MUST be valid

- JavaScript MUST:
  - work in browser
  - avoid syntax errors
  - manipulate DOM correctly

- React projects MUST:
  - include valid imports
  - export default components
  - avoid duplicate components
  - avoid mixing multiple files

- NEVER leave placeholder code
- NEVER leave TODO comments
- Generate COMPLETE production-ready code
- Ensure syntax is fully valid

IMPORTANT:
Generate ONLY the code for:
{current_task.filepath}
"""

    # =====================================
    # GENERATE CODE
    # =====================================

    try:

        response = llm.invoke(prompt)

        generated_code = clean_code(
            response.content
        )

        # =====================================
        # VALIDATE EMPTY OUTPUT
        # =====================================

        if not generated_code.strip():

            generated_code = (
                f"// Failed to generate content "
                f"for {current_task.filepath}"
            )

        # =====================================
        # SAVE FILE
        # =====================================

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

    # =====================================
    # NEXT STEP
    # =====================================

    coder_state.current_step_idx += 1

    return {
        "coder_state": coder_state,
    }

# =====================================
# BUILD GRAPH
# =====================================

graph = StateGraph(dict)

# =====================================
# NODES
# =====================================

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

# =====================================
# FLOW
# =====================================

graph.add_edge(
    "planner",
    "architect"
)

graph.add_edge(
    "architect",
    "coder"
)

# =====================================
# LOOP CODER UNTIL DONE
# =====================================

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

# =====================================
# ENTRY POINT
# =====================================

graph.set_entry_point(
    "planner"
)

# =====================================
# COMPILE GRAPH
# =====================================

agent = graph.compile()

# =====================================
# LOCAL TESTING
# =====================================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "user_prompt": (
                "Build a colourful modern "
                "todo app using html css "
                "and javascript"
            )
        },
        {
            "recursion_limit": 20
        }
    )

    print(result)
