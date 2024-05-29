import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
from langchain import hub
dotenv.load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# 获取网页信息
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# url = "https://blog.web-of-anion.top/archives/langchainxue-xi-bi-ji"

# Only keep post title, headers, and content from the full HTML.
# 获取特定部分
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# 加载器
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs={"parse_only": bs4_strainer},
)
# 使用加载器加载内容并存储在docs中
docs = loader.load()
docs[0].page_content = docs[0].page_content[:5000]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# all_splits = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# ).split_documents(WebBaseLoader(web_paths=(url,),bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))},).load())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 存进vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 实例化检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 用检索器检索问题的答案
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# with open('output.txt', 'w', encoding='utf-8') as f:
#     for para in retrieved_docs:
#         print(para, file=f, end="\n\n\n")

from langchain import hub

# 加载预设rag模板


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-turbo")


# 定义chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 实际上这里就分叉了，参考runnableparallel
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# question = input("input your question:")
# for chunk in rag_chain.stream(question):
#     print(chunk, end="", flush=True)