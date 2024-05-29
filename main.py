import os
from modules import load_prompt
import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()
store = {}



def get_session_history(session_id: str) :
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def filter_messages(messages, k=10):
    return messages[-k:]


model = ChatOpenAI(model="gpt-4-turbo")

system_template = load_prompt.load_from_file("system_prompt.txt")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "hello"),
        MessagesPlaceholder(variable_name="messages"),
    ]

)

chain = (RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"])) |
         prompt_template | model | StrOutputParser())

# 自定义的RunnablePassthrough来处理字典输入
class ContextExtractor(RunnablePassthrough):
    def invoke(self, input, config):
        # 检查输入是否为字典并包含'context'键
        if isinstance(input, dict) and 'context' in input:
            return input['context']
        else:
            raise ValueError("输入必须是包含'context'键的字典")

    def stream(self, input):
        # Stream版本的invoke方法
        if isinstance(input, dict) and 'context' in input:
            yield input['context']
        else:
            raise ValueError("输入必须是包含'context'键的字典")

from modules.rag import retriever, format_docs
ch = ContextExtractor()
# print(ch.invoke({"context": "what is rag"}))
# tp_ch = RunnablePassthrough.assign(messages=lambda x: x["context"]) | retriever | format_docs | StrOutputParser()
# print(tp_ch.invoke({"context": "what is rag"}))


# 定义chain
rag_chain = (
    {
        "context": ContextExtractor() | retriever | format_docs,
        "messages": (lambda x: filter_messages(x["messages"]))
    }  # 实际上这里就分叉了，参考runnableparallel
    | prompt_template
    | model
    | StrOutputParser()
)

with_message_history_chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="messages"
)


def get_completion(user_prompt, session_id):

    for r in with_message_history_chatbot.stream(
        {
            "context": user_prompt,
            "messages": [HumanMessage(content=user_prompt)],
            "job": load_prompt.load_from_file("job.txt"),
            "job_desc": load_prompt.load_from_file("job_desc.txt"),
            "job_req": load_prompt.load_from_file("job_req.txt")
        },
        config={"configurable": {"session_id": session_id}}
    ):
        yield r


def run_console_chat():
    print("欢迎使用聊天机器人！输入'退出'以结束对话。")
    while True:
        user_prompt = input("你: ")
        if user_prompt.lower() in ["退出", "exit"]:
            print("聊天结束，再见！")
            break
        print("机器人：", end="")
        # for token in get_completion(user_prompt, "first_session"):
        #     print(token, end="")
        print(with_message_history_chatbot.invoke({
                "context": user_prompt,
                "messages": [HumanMessage(content=user_prompt)],
                "job": load_prompt.load_from_file("job.txt"),
                "job_desc": load_prompt.load_from_file("job_desc.txt"),
                "job_req": load_prompt.load_from_file("job_req.txt")
            },
            config={"configurable": {"session_id": "123"}}
            )
        )
        print()


# 启动控制台聊天
run_console_chat()



