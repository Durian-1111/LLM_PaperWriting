import os
print('工作路径',os.getcwd())
print('当前目录',os.path.abspath(os.curdir))
os.system("python -m pip install --upgrade pip;pip install modelscope==1.9.5;pip install transformers==4.35.2;pip install streamlit==1.24.0;pip install sentencepiece==0.1.99;pip install accelerate==0.24.1;pip install chromadb==0.4.15;pip install sentence-transformers==2.2.2;pip install unstructured==0.10.30;pip install markdown==3.3.7")


os.system("pip install langchain==0.0.292")
os.system("git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages;cd nltk_data;mv packages/*  ./;cd tokenizers;unzip punkt.zip;cd ../taggers;unzip averaged_perceptron_tagger.zip")


#download model
if not os.path.exists('/home/xlab-app-center/InternLM-chat-7b-8k'):
    os.system("pip install -U openxlab")
    from openxlab.model import download
    download(model_repo='OpenLMLab/InternLM-chat-7b-8k',output='/home/xlab-app-center/InternLM-chat-7b-8k')
    os.system("python langchain/LLM.py")
    # YuanLLM/Yuan2-2B-hf
# download(model_repo='YuanLLM/Yuan2-2B-hf',output='/home/xlab-app-center/InternLM-chat-7b-8k')
    # os.system("pip install flash_attn")
if not os.path.exists('/home/xlab-app-center/sentence-transformer'):
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 下载模型
    os.system('pip install -U huggingface_hub; huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /home/xlab-app-center/sentence-transformer')


os.system("pip install -r langchain/requirements.txt")
os.system("pip install chromadb==0.3.29;pip install opencv-python;pip install pytesseract;pip install python-docx;pip install -U pypdf;")#pip install chromadb==0.3.29;
os.system("pip install datasets==2.12.0;pip install gradio==3.37.0;pip install matplotlib==3.7.2;pip install numpy==1.24.4;pip install peft==0.5.0;")
os.system("python langchain/create_db.py")
# 导入必要的库
print(os.getcwd())
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/home/xlab-app-center/sentence-transformer")
    # 向量数据库持久化路径
    persist_directory = '/home/xlab-app-center/math_base'
    model_path='/home/xlab-app-center/InternLM-chat-7b-8k'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    llm = InternLM_LLM(model_path = '/home/xlab-app-center/InternLM-chat-7b-8k')

    template = """ 回答时需要遵循以下用---括起来的格式：
                   可参考的解题思路：
                    ···
                    {context}
                    ···
                    ---
                    Question: 需要回答的问题。
                    Thought: 拆解问题，将问题中所有的概念解释一遍。
                    record: 记录Thought要拆解的问题，把问题抽象为公式。 
                    answer:  根据recode的步骤，逐条回答record的问题。
                    Observation: 回看所有步骤，验证并回答问题，请将每一步都详细的解答。
                    Thought: 我现在知道最终答案。如果不太确定，可以重复多次Thought,record,answer,Observation
                    ...（这个思考/行动/行动输入/观察可以重复N次）
                    Final Answer: 原始输入问题的最终答案
                    ---
                    现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。                     
                    使用以下上下文来回答用户的问题。如果你不知道答案，则重复Thought步骤。
                    Question: {question}
                    
                    重复record,answer步骤2次。总是使用中文回答。
                    验证过的回答:
                    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)

    # 运行 chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

class Model_center():
    """
    存储问答 Chain 的对象 
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            print(chat_history)
            return "", chat_history
        except Exception as e:
            return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
