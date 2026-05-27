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

    return "\n".join(cleaned_lines).strip()

# =====================================
# PLANNER AGENT
# =====================================

def planner_agent(state: dict) -> dict:
    """
    Convert user prompt into structured Plan.
    """

    user_prompt = state["user_prompt"]

    response = llm.with_structured_output(
        Plan
    ).invoke(
        planner_prompt(user_prompt)
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
- Generate code ONLY for this exact file:
  {current_task.filepath}

- NEVER generate code for other files
- NEVER include multiple files in one response
- NEVER include filenames in output
- NEVER include markdown
- NEVER include ``` fences
- NEVER explain anything
- Return ONLY raw code
- Generate COMPLETE valid code
- Ensure imports/exports are correct
- Ensure syntax is valid
- Ensure production-ready code

IMPORTANT:
Your response MUST contain code for ONLY:
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

# Nodes
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
