import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import prenatal_agent

st.set_page_config(page_title="Prenatal Assistant", page_icon="🤰")
st.title("Third-Trimester Prenatal Assistant")
st.caption("Week 28–31 Support • Kick Counts, Nutrition & Checklists")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about symptoms, kick counts, or your hospital bag..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": "iphone_user_01"}}
    final_reply = ""
    for event in prenatal_agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="values",
    ):
        if "messages" in event:
            last = event["messages"][-1]
            if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                final_reply = last.content

    st.session_state.messages.append({"role": "assistant", "content": final_reply})
    with st.chat_message("assistant"):
        st.markdown(final_reply)
