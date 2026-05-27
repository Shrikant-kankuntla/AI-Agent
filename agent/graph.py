from dotenv import load_dotenv

from langchain.globals import (
    set_verbose,
    set_debug,
)

from langchain_groq import ChatGroq

from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.prompts import (
    planner_prompt,
    architect_prompt,
)

from agent.states import (
    Plan,
    TaskPlan,
    CoderState,
)

from agent.tools import (
    write_file,
    read_file,
)

# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

# ==========================================
# DISABLE LOGS
# ==========================================

set_debug(False)
set_verbose(False)

# ==========================================
# FAST + STABLE MODEL
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

    bad_prefixes = [
        "```html",
        "```css",
        "```javascript",
        "```js",
        "```python",
        "```",
    ]

    for prefix in bad_prefixes:
        content = content.replace(prefix, "")

    return content.strip()

# ==========================================
# VALIDATION
# ==========================================

def validate_code(filepath: str, content: str) -> bool:

    if not content.strip():
        return False

    if "```" in content:
        return False

    # HTML validation
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

    # CSS validation
    elif filepath.endswith(".css"):

        if "{" not in content:
            return False

        if "}" not in content:
            return False

    # JS validation
    elif filepath.endswith(".js"):

        bad_patterns = [
            "TODO",
            "lorem ipsum",
        ]

        for pattern in bad_patterns:

            if pattern.lower() in content.lower():
                return False

    return True

# ==========================================
# PLANNER AGENT
# ==========================================

def planner_agent(state: dict) -> dict:

    user_prompt = state["user_prompt"]

    enhanced_prompt = f"""
{planner_prompt(user_prompt)}

IMPORTANT RULES:

- For simple apps use:
  html, css, javascript

- Use React ONLY if user explicitly asks

- Avoid backend unless requested

- Prefer minimal file structures
"""

    response = llm.with_structured_output(
        Plan
    ).invoke(
        enhanced_prompt
    )

    if response is None:

        raise ValueError(
            "Planner failed."
        )

    return {
        "plan": response
    }

# ==========================================
# ARCHITECT AGENT
# ==========================================

def architect_agent(state: dict) -> dict:

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
            "Architect failed."
        )

    response.plan = plan

    return {
        "task_plan": response
    }

# ==========================================
# CODER AGENT
# ==========================================

def coder_agent(state: dict) -> dict:

    coder_state: CoderState = state.get(
        "coder_state"
    )

    # ======================================
    # INIT STATE
    # ======================================

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

    # ======================================
    # STOP CONDITION
    # ======================================

    if coder_state.current_step_idx >= len(steps):

        return {
            "coder_state": coder_state,
            "status": "DONE",
        }

    # ======================================
    # CURRENT TASK
    # ======================================

    current_task = steps[
        coder_state.current_step_idx
    ]

    # ======================================
    # EXISTING CONTENT
    # ======================================

    try:

        existing_content = read_file.invoke(
            {
                "path": current_task.filepath
            }
        )

    except Exception:

        existing_content = ""

    # ======================================
    # PROMPT
    # ======================================

    prompt = f"""
You are a senior software engineer.

Generate COMPLETE code for ONE file only.

PROJECT:
{coder_state.task_plan.plan.model_dump_json()}

TASK:
{current_task.task_description}

TARGET FILE:
{current_task.filepath}

EXISTING CONTENT:
{existing_content}

STRICT RULES:
- Generate ONLY raw code
- No markdown
- No explanations
- No code fences
- No extra text
- Generate COMPLETE file
- Ensure valid syntax
- Ensure production-ready code
- NEVER generate multiple files
- NEVER mention filenames
"""

    # ======================================
    # GENERATE CODE
    # ======================================

    generated_code = ""

    try:

        max_retries = 3

        for attempt in range(max_retries):

            response = llm.invoke(prompt)

            generated_code = clean_code(
                response.content
            )

            if validate_code(
                current_task.filepath,
                generated_code,
            ):
                break

        # ==================================
        # FALLBACK CONTENT
        # ==================================

        if not generated_code.strip():

            if current_task.filepath.endswith(".html"):

                generated_code = """
<!DOCTYPE html>
<html>
<head>
    <title>Generated App</title>
</head>
<body>
    <h1>Generated App</h1>
</body>
</html>
"""

            elif current_task.filepath.endswith(".css"):

                generated_code = """
body {
    font-family: Arial;
}
"""

            elif current_task.filepath.endswith(".js"):

                generated_code = """
console.log("Generated App");
"""

            else:

                generated_code = "Generated file"

        # ==================================
        # SAVE FILE
        # ==================================

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

    # ======================================
    # NEXT STEP
    # ======================================

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

# ==========================================
# LOOP
# ==========================================

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
# LOCAL TEST
# ==========================================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "user_prompt": (
                "Build a colourful calculator "
                "using html css and javascript"
            )
        },
        {
            "recursion_limit": 20
        }
    )

    print(result)
