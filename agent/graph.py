from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import *
from agent.states import *
from agent.tools import (
    write_file,
    read_file,
    get_current_directory,
    list_files,
)

# =========================
# Load environment
# =========================

load_dotenv()

# Disable noisy logs in Streamlit
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
    Converts user prompt into structured Plan.
    """

    user_prompt = state["user_prompt"]

    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )

    if resp is None:
        raise ValueError("Planner failed to generate plan.")

    return {"plan": resp}


# =========================
# Architect Agent
# =========================

def architect_agent(state: dict) -> dict:
    """
    Converts Plan into TaskPlan.
    """

    plan: Plan = state["plan"]

    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )

    if resp is None:
        raise ValueError("Architect failed to generate task plan.")

    resp.plan = plan

    return {"task_plan": resp}


# =========================
# Coder Agent
# =========================

def coder_agent(state: dict) -> dict:
    """
    Tool-using coding agent.
    """

    coder_state: CoderState = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = coder_state.task_plan.implementation_steps

    # Stop condition
    if coder_state.current_step_idx >= len(steps):
        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    current_task = steps[coder_state.current_step_idx]

    # Read existing file safely
    try:
        existing_content = read_file.invoke(
            {"path": current_task.filepath}
        )
    except Exception:
        existing_content = ""

    # Strong anti-hallucination instruction
    system_prompt = f"""
You are a coding agent.

You MUST ONLY use the following tools:
- read_file
- write_file
- list_files
- get_current_directory

DO NOT invent tools.
DO NOT call tools that are not provided.

Always use write_file to save code changes.
"""

    user_prompt = f"""
Task:
{current_task.task_description}

File:
{current_task.filepath}

Existing Content:
{existing_content}

Implement the requested changes carefully.
"""

    coder_tools = [
        read_file,
        write_file,
        list_files,
        get_current_directory,
    ]

    react_agent = create_react_agent(
        llm,
        tools=coder_tools,
    )

    try:
        react_agent.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
            }
        )

    except Exception as e:
        return {
            "coder_state": coder_state,
            "status": f"ERROR: {str(e)}",
        }

    coder_state.current_step_idx += 1

    return {
        "coder_state": coder_state,
    }


# =========================
# Graph
# =========================

graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {
        "END": END,
        "coder": "coder",
    },
)

graph.set_entry_point("planner")

agent = graph.compile()

# =========================
# Local testing
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
            "recursion_limit": 100
        }
    )

    print(result)
