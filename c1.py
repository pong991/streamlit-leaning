import streamlit as st
# 导入 LangChain 的相关模块，用于处理对话和使用 OpenAI 模型
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatZhipuAI


# 使用 Streamlit 设置页面配置，如页面标题
st.set_page_config(page_title="Conversational Q&A Chatbot")
# 创建页面头部标题
st.header("Hey, Let's chat") #############################################################

from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 在代码中使用环境变量
import os
ZHIPUAI_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化 LangChain 中的 ChatZhipuAI 类，设置对话模型的温度参数为 0.5
chat = ChatZhipuAI(
    api_key=ZHIPUAI_API_KEY,
    model="glm-4",
    temperature=0.5,
)

# 判断会话状态中是否有 "FlowMessages"，如果没有，则初始化它
if "FlowMessages" not in st.session_state:
    st.session_state["FlowMessages"] = [
        SystemMessage(content="You are an AI assistant"),
    ]

# 定义一个函数，用于获取用户输入的问题，并返回模型的回答
def get_chatmodel_responses(question):
    """Get a prompt from user and return a response/answer."""

    # 将用户问题添加到会话状态中
    st.session_state["FlowMessages"].append(HumanMessage(content=question))


    # 将问题传递给我们的 OpenAI 模型
    answer = chat.invoke(st.session_state["FlowMessages"])

    # 将模型的回答也添加到会话状态中 ########### 从演示来看，app既记不住历史聊天记录，也无法展示为多轮对话
    st.session_state["FlowMessages"].append(AIMessage(content=answer.content))

    # 返回模型的回答内容
    return answer.content

# 创建一个文本输入框，用户可以在其中输入问题 ######################################################
input = st.text_input("Input: ", key="input")
# 调用 get_chatmodel_responses 函数，获取回答
if input:
    response = get_chatmodel_responses(input)

# 创建一个按钮，当用户点击时，显示问题的回答
submit = st.button("Ask the question")

# 如果点击了“Ask the question”按钮
if submit:
    # 显示子标题“回答是”
    st.subheader("The Response is") ##########################################################
    # 显示模型的回答
    st.write(response) #############################################################