import streamlit as st
from Prompt import chatbot_chain,report_chain,upsert_embedding
from database import *

response=None
products=''

st.header("Customer Assist Chat-Bot ")
x=st.write("Whats on your mind")

if "messages" not in st.session_state:
	st.session_state.messages=[]
	

for message in st.session_state.messages:
	with st.chat_message(message["role"],avatar= "👤" if message["role"]=="user" else "🤖"):
		st.markdown(message["message"])


if query:=st.chat_input("Enter your message"):
	with st.chat_message("human",avatar="👤"):
		st.markdown(query)
	st.session_state.messages.append({"role":"user","message":query})
	response=chatbot_chain.invoke({"chats":st.session_state.messages,"query":query,'products':products,'past_info':past_info})

if response:
	with st.chat_message("ai",avatar="🤖"):
		st.markdown(response)
	st.session_state.messages.append({"role":"ai","message":response})
	response=None

if st.button("End Session"):
	st.write("Session ended Thank You!!!")
	session_data=[]
	for i in st.session_state.messages:
		session_data.append(i["role"]+":"+i["message"]+"\n")
	vectordata=report_chain.invoke({'session_data':session_data })
	st.write(f"vector data :{vectordata}")
	upsert_embedding(vectordata);