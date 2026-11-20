import os
from datetime import date, datetime
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from db import get_conn, ensure_user, init_db

load_dotenv()
init_db()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

RED_FLAGS = [
    "bleeding", "heavy bleeding", "spotting",
    "severe headache", "blurry vision", "vision changes",
    "chest pain", "shortness of breath", "difficulty breathing",
    "water broke", "leaking fluid", "gush of fluid",
    "reduced fetal movement", "baby stopped moving", "no kicks",
    "severe abdominal pain", "regular contractions"
]

MEDICAL_DISCLAIMER = (
    "\n\n---\n*Note: I am an AI assistant, not a healthcare provider. "
    "Always confirm symptom or medication decisions with your OB/GYN or midwife.*"
)

@tool
def calculate_gestational_age(config: RunnableConfig) -> str:
    """Calculates current gestational week and days based on the user's saved due date."""
    user_id = (config or {}).get("configurable", {}).get("user_id", "default_user")
    ensure_user(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT due_date FROM users WHERE user_id = %s;", (user_id,))
            row = cur.fetchone()
    due_date = row["due_date"]
    today = date.today()
    days_remaining = (due_date - today).days
    total_days_pregnant = 280 - days_remaining
    weeks = max(0, total_days_pregnant // 7)
    days = max(0, total_days_pregnant % 7)
    return (
        f"Due date: {due_date.isoformat()}. "
        f"Current gestational age: {weeks} weeks and {days} days ({days_remaining} days remaining)."
    )

@tool
def log_kick_count(movements: int, minutes_elapsed: int, config: RunnableConfig) -> str:
    """Logs a fetal kick count session (movements counted and minutes elapsed) to the database."""
    user_id = (config or {}).get("configurable", {}).get("user_id", "default_user")
    ensure_user(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kick_logs (user_id, movements, minutes_elapsed)
                VALUES (%s, %s, %s);
                """,
                (user_id, movements, minutes_elapsed),
            )
            cur.execute(
                """
                SELECT movements, minutes_elapsed, logged_at
                FROM kick_logs
                WHERE user_id = %s
                ORDER BY logged_at DESC
                LIMIT 5;
                """,
                (user_id,),
            )
            recent = cur.fetchall()
        conn.commit()
    status = "Reassuring (10+ movements within 2 hours)." if movements >= 10 and minutes_elapsed <= 120 else "Below 10 movements—recommend resting on your left side, hydrating, and re-counting or calling your OB/GYN."
    history = "; ".join(f"{r['movements']} kicks in {r['minutes_elapsed']}m" for r in recent)
    return f"Logged {movements} movements in {minutes_elapsed} minutes. {status} Recent logs: {history}"

@tool
def manage_checklist(
    action: Literal["view", "add", "remove"],
    category: Literal["hospital_bag", "nursery", "postpartum"],
    item: str = "",
    config: RunnableConfig = None,
) -> str:
    """Views, adds, or removes items from the user's persistent birth-prep checklist."""
    user_id = (config or {}).get("configurable", {}).get("user_id", "default_user")
    ensure_user(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if action == "add" and item.strip():
                cur.execute(
                    """
                    INSERT INTO checklists (user_id, category, item)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, category, item) DO NOTHING;
                    """,
                    (user_id, category, item.strip()),
                )
            elif action == "remove" and item.strip():
                cur.execute(
                    """
                    DELETE FROM checklists
                    WHERE user_id = %s AND category = %s AND LOWER(item) = LOWER(%s);
                    """,
                    (user_id, category, item.strip()),
                )
            cur.execute(
                """
                SELECT item FROM checklists
                WHERE user_id = %s AND category = %s
                ORDER BY id ASC;
                """,
                (user_id, category),
            )
            items = [r["item"] for r in cur.fetchall()]
        conn.commit()
    if not items:
        return f"Your {category.replace('_', ' ')} checklist is currently empty."
    return f"{category.replace('_', ' ').title()} checklist: {', '.join(items)}"

TOOLS = [calculate_gestational_age, log_kick_count, manage_checklist]
llm = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a warm, supportive third-trimester prenatal assistant. "
    "Use your tools to check gestational age, log kick counts, and manage birth preparation checklists. "
    "Always remind the user to consult their OB/GYN or midwife for medical decisions."
)

def safety_triage(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    if any(flag in last_msg for flag in RED_FLAGS):
        urgent = (
            "**URGENT MEDICAL ALERT:** The symptom you mentioned can be a warning sign in the third trimester. "
            "Please contact your OB/GYN, midwife, or Labor & Delivery triage immediately—or go to the nearest emergency room."
            + MEDICAL_DISCLAIMER
        )
        return {"messages": [AIMessage(content=urgent)]}
    return {}

def route_after_triage(state: AgentState):
    if isinstance(state["messages"][-1], AIMessage):
        return END
    return "agent_node"

def agent_node(state: AgentState):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    if not response.tool_calls and response.content:
        if "not a healthcare provider" not in response.content:
            response.content += MEDICAL_DISCLAIMER
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("safety_triage", safety_triage)
workflow.add_node("agent_node", agent_node)
workflow.add_node("tools", ToolNode(TOOLS))

workflow.set_entry_point("safety_triage")
workflow.add_conditional_edges("safety_triage", route_after_triage, {"agent_node": "agent_node", END: END})
workflow.add_conditional_edges("agent_node", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent_node")

prenatal_agent = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    print("Prenatal Assistant Ready (Type 'quit' to exit)")
    cfg = {"configurable": {"thread_id": "cli_test", "user_id": "ermi_test"}}
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "quit":
            break
        result = prenatal_agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=cfg)
        print("\nAssistant:\n" + result["messages"][-1].content)
