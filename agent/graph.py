from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.prompts import *
from agent.states import *
from agent.tools import (
    write_file,
    read_file,
)

# =========================
# Load environment
# =========================

load_dotenv()

# Disable noisy logs
set_debug(False)
set_verbose(False)

# =========================
# LLM
# =========================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# =========================
# Planner Agent
# =========================

def planner_agent(state: dict) -> dict:
    """
    Converts user prompt into a structured project plan.
    """

    user_prompt = state["user_prompt"]

    response = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )

    if response is None:
        raise ValueError("Planner failed to generate plan.")

    return {
        "plan": response
    }


# =========================
# Architect Agent
# =========================

def architect_agent(state: dict) -> dict:
    """
    Converts Plan into TaskPlan.
    """

    plan: Plan = state["plan"]

    response = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(
            plan=plan.model_dump_json()
        )
    )

    if response is None:
        raise ValueError("Architect failed to generate task plan.")

    response.plan = plan

    return {
        "task_plan": response
    }


# =========================
# Coder Agent
# =========================

def coder_agent(state: dict) -> dict:
    """
    Generates code file-by-file.
    """

    coder_state: CoderState = state.get("coder_state")

    # Initialize coder state
    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = coder_state.task_plan.implementation_steps

    # =========================
    # Stop condition
    # =========================

    if coder_state.current_step_idx >= len(steps):
        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    current_task = steps[coder_state.current_step_idx]

    # =========================
    # Read existing content
    # =========================

    try:
        existing_content = read_file.invoke(
            {
                "path": current_task.filepath
            }
        )
    except Exception:
        existing_content = ""

    # =========================
    # Prompt
    # =========================

    prompt = f"""
You are an expert software engineer.

Generate COMPLETE production-ready code.

Task:
{current_task.task_description}

File Path:
{current_task.filepath}

Existing Content:
{existing_content}

IMPORTANT RULES:
- Return ONLY raw code
- No markdown
- No explanations
- No code fences
- Generate FULL file content
- Preserve existing functionality if present
"""

    # =========================
    # Generate code
    # =========================

    try:

        response = llm.invoke(prompt)

        generated_code = response.content.strip()

        # Save file
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

    # =========================
    # Move to next step
    # =========================

    coder_state.current_step_idx += 1

    return {
        "coder_state": coder_state,
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

# Flow
graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

# Loop coder until done
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

# Entry point
graph.set_entry_point("planner")

# Compile graph
agent = graph.compile()

# =========================
# Local Testing
# =========================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "user_prompt": (
                "Build a colourful modern todo app "
                "using html css and javascript"
            )
        },
        {
            "recursion_limit": 20
        }
    )

    print(result)
