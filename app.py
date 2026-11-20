import uuid
from datetime import date
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import prenatal_agent
from db import ensure_user, get_conn, save_message, load_messages, clear_messages

st.set_page_config(page_title="Prenatal Assistant", page_icon="🤰")
st.title("Third-Trimester Prenatal Assistant")
st.caption("Week 28–31 Support • Kick Counts, Nutrition & Checklists")

with st.sidebar:
    st.header("Your Profile")
    user_id = st.text_input("Profile ID or Email", value="default_user").strip() or "default_user"
    due_date = st.date_input("Estimated Due Date", value=date(2027, 2, 5))
    if st.button("Save Profile"):
        ensure_user(user_id, due_date.isoformat())
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET due_date = %s WHERE user_id = %s;",
                    (due_date.isoformat(), user_id),
                )
            conn.commit()
        st.success("Profile & due date saved!")

    st.divider()
    if st.button("Clear Chat History"):
        clear_messages(user_id)
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

ensure_user(user_id, due_date.isoformat())

if "active_user" not in st.session_state or st.session_state.active_user != user_id:
    st.session_state.active_user = user_id
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = load_messages(user_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about symptoms, kick counts, or your hospital bag..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(user_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    history_msgs = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[-12:]
    ]

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": user_id,
        }
    }
    final_reply = ""
    for event in prenatal_agent.stream(
        {"messages": history_msgs},
        config=config,
        stream_mode="values",
    ):
        if "messages" in event:
            last = event["messages"][-1]
            if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                final_reply = last.content

    if final_reply:
        st.session_state.messages.append({"role": "assistant", "content": final_reply})
        save_message(user_id, "assistant", final_reply)
        with st.chat_message("assistant"):
            st.markdown(final_reply)
