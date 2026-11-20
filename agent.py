import os
import re
from datetime import datetime, date
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

USER_DB = {
    "due_date": "2027-02-05",
    "kick_logs": [],
    "checklists": {
        "hospital_bag": ["Photo ID & insurance card", "Comfortable going-home outfit"],
        "nursery": ["Crib assembly", "Install infant car seat"],
        "doctor_questions": ["Ask about Tdap vaccine timing"],
    },
}

CLINICAL_KB = {
    "kick_counts": (
        "ACOG Guidelines (Third Trimester): Starting around week 28, monitor fetal movements daily. "
        "Choose a time when baby is usually active. Time how long it takes to feel 10 distinct movements "
        "(kicks, flutters, rolls). Ideally, you should feel 10 movements within 2 hours."
    ),
    "nutrition": (
        "Third Trimester Nutrition: Focus on iron-rich foods (lean beef, spinach, lentils) to support "
        "increased blood volume, calcium (1,000 mg/day), and DHA. Eat smaller, more frequent meals to "
        "alleviate third-trimester heartburn caused by upward pressure on the stomach."
    ),
    "braxton_hicks": (
        "Braxton Hicks vs. Preterm Labor: Braxton Hicks are irregular, mild tightening sensations that "
        "usually stop when you change positions or hydrate. Regular, painful contractions occurring every "
        "10 minutes or less before 37 weeks require immediate medical evaluation."
    ),
    "vaccines": (
        "Third Trimester Screenings & Vaccines: Between weeks 27 and 36, the Tdap vaccine is recommended "
        "to pass whooping cough antibodies to the baby. Your provider will also monitor blood pressure "
        "and urine protein at each bi-weekly visit."
    ),
}

@tool
def calculate_gestational_age() -> str:
    """Calculates the user's current gestational week and day based on their stored due date."""
    due = datetime.strptime(USER_DB["due_date"], "%Y-%m-%d").date()
    today = date.today()
    days_remaining = (due - today).days
    total_gestation_days = 280
    days_pregnant = total_gestation_days - days_remaining

    if days_pregnant < 0:
        return "Due date appears to be more than 40 weeks away. Please verify due date."

    weeks = days_pregnant // 7
    days = days_pregnant % 7
    return f"Current gestational age: {weeks} weeks and {days} days (Due Date: {USER_DB['due_date']})."

@tool
def log_kick_count(movements: int, minutes_elapsed: int) -> str:
    """Logs a fetal kick count session and evaluates against standard 10-movements-in-2-hours guidance."""
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "movements": movements,
        "minutes": minutes_elapsed,
    }
    USER_DB["kick_logs"].append(entry)

    if movements >= 10 and minutes_elapsed <= 120:
        return (
            f"Logged: {movements} movements in {minutes_elapsed} minutes. "
            "This meets the standard reassuring milestone of 10 movements within 2 hours."
        )
    elif minutes_elapsed >= 120 and movements < 10:
        return (
            f"ALERT LOGGED: Only {movements} movements recorded in {minutes_elapsed} minutes. "
            "Because this is fewer than 10 movements in 2 hours, please drink cold water, lie on your "
            "left side, and contact your OB/GYN or midwife right away for guidance."
        )
    else:
        return (
            f"Logged: {movements} movements in {minutes_elapsed} minutes. "
            "Keep counting until you reach 10 movements or 2 hours have passed."
        )

@tool
def manage_checklist(
    action: Literal["view", "add", "remove"],
    category: Literal["hospital_bag", "nursery", "doctor_questions"],
    item: str = "",
) -> str:
    """Views, adds, or removes items from the user's third-trimester preparation checklists."""
    target_list = USER_DB["checklists"].get(category, [])

    if action == "view":
        formatted = "\n".join(f"- {i}" for i in target_list) if target_list else "Empty"
        return f"Checklist ({category}):\n{formatted}"

    if action == "add" and item:
        target_list.append(item)
        return f"Added '{item}' to {category} checklist."

    if action == "remove" and item:
        if item in target_list:
            target_list.remove(item)
            return f"Removed '{item}' from {category} checklist."
        return f"Item '{item}' not found in {category}."

    return "Invalid checklist operation."

@tool
def search_clinical_guidelines(topic: Literal["kick_counts", "nutrition", "braxton_hicks", "vaccines"]) -> str:
    """Retrieves verified third-trimester clinical guidelines for a specific topic."""
    return CLINICAL_KB.get(topic, "No specific guideline found. Advise consulting OB/GYN.")

tools = [
    calculate_gestational_age,
    log_kick_count,
    manage_checklist,
    search_clinical_guidelines,
]
tools_by_name = {t.name: t for t in tools}

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    is_emergency: bool

EMERGENCY_PATTERNS = [
    r"\bbleed(ing)?\b",
    r"\bspot(ting)?\b",
    r"\bfluid leak(ing)?\b",
    r"\bwater broke\b",
    r"\bblurred vision\b",
    r"\bvision changes\b",
    r"\bspots in vision\b",
    r"\bsevere headache\b",
    r"\bsudden swelling\b",
    r"\bface swell(ing)?\b",
    r"\bcan'?t feel (the )?baby\b",
    r"\bno movement\b",
    r"\bdecreased movement\b",
    r"\bsevere pain\b",
    r"\bchest pain\b",
    r"\bshortness of breath\b",
]

def safety_triage_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic pre-LLM router that catches high-risk third-trimester symptoms."""
    last_message = state["messages"][-1].content.lower()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, last_message):
            escalation_msg = (
                "🚨 **MEDICAL SAFETY ALERT** 🚨\n\n"
                "The symptom you mentioned requires immediate medical evaluation during the third trimester. "
                "**Please contact your OB/GYN, midwife, or Labor & Delivery triage unit right away.**\n\n"
                "If you are experiencing severe pain, heavy bleeding, vision changes, or difficulty breathing, "
                "seek urgent emergency care immediately. I cannot evaluate emergency symptoms."
            )
            return {
                "messages": [AIMessage(content=escalation_msg)],
                "is_emergency": True,
            }

    return {"is_emergency": False}

def route_after_triage(state: AgentState) -> Literal["agent_node", END]:
    if state.get("is_emergency"):
        return END
    return "agent_node"

SYSTEM_PROMPT = """You are a warm, calm, and highly organized prenatal assistant supporting a woman in her 7th month of pregnancy (Third Trimester, Weeks 28–31).

Core Directives:
1. Always call `calculate_gestational_age` on your first interaction or whenever pregnancy timing is relevant so your advice matches her exact week.
2. Keep responses concise, empathetic, and structured with clear bullet points. Avoid overwhelming walls of text.
3. Use `search_clinical_guidelines` before answering health, nutrition, kick count, or contraction questions.
4. NEVER diagnose conditions, interpret lab results, or approve over-the-counter/prescription medications (e.g., NSAIDs/ibuprofen). Always defer medication questions to her OB/GYN or pharmacist.
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0.2).bind_tools(tools)

def agent_node(state: AgentState) -> Dict[str, Any]:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    tool_outputs = []
    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_outputs.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": tool_outputs}

def should_continue(state: AgentState) -> Literal["tool_node", "output_guardrail"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "output_guardrail"

def output_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """Post-LLM guardrail appending a brief medical disclaimer when physical symptoms/meds are discussed."""
    last_msg = state["messages"][-1]
    content = last_msg.content

    medical_keywords = ["symptom", "pain", "contraction", "medication", "ibuprofen", "tylenol", "heartburn", "swelling", "kick"]
    if any(kw in content.lower() for kw in medical_keywords):
        disclaimer = "\n\n*Note: I am an AI assistant, not a healthcare provider. Always confirm symptom or medication decisions with your OB/GYN or midwife.*"
        if disclaimer.strip() not in content:
            last_msg.content = content + disclaimer

    return {"messages": [last_msg]}

workflow = StateGraph(AgentState)

workflow.add_node("safety_triage", safety_triage_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("output_guardrail", output_guardrail_node)

workflow.add_edge(START, "safety_triage")
workflow.add_conditional_edges("safety_triage", route_after_triage)
workflow.add_conditional_edges("agent_node", should_continue)
workflow.add_edge("tool_node", "agent_node")
workflow.add_edge("output_guardrail", END)

memory = MemorySaver()
prenatal_agent = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_session_01"}}
    print("Prenatal Assistant Ready (Type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        events = prenatal_agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                last = event["messages"][-1]
                if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                    final_reply = last.content

        print(f"\nAssistant:\n{final_reply}\n" + "-" * 50)
