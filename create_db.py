# 首先导入所需第三方库
import os
print('db工作路径',os.getcwd())
print('db当前目录',os.path.abspath(os.curdir))
os.system("pip install pytesseract;pip install python-docx;pip install docx2txt")
# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
 
 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import re

model_name = "/home/xlab-app-center/sentence-transformer"
# 定义持久化路径
persist_directory = '/home/xlab-app-center/math_base'
# 目标文件夹
tar_dir = [
    "/home/xlab-app-center/langchain/files"
]
# 规范文件名 避免报错
pat = re.compile(r'[a-z0-9\u4e00-\u9fa5]+')
def rename_file(file_dir):
    file_name = os.path.basename(file_dir)
    _ix = file_name.split('.')[-1]
    file_name = file_name.split('.')[0]
    new_filename = ''.join(re.findall(pat,file_name))+'.'+_ix
    os.rename(file_dir, os.path.join(os.path.dirname(file_dir), new_filename))
    return os.path.join(os.path.dirname(file_dir), new_filename)
# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            # 如果满足要求，将其绝对路径加入到结果列表
            # if filename.endswith(".md"):
            #     file_dir = os.path.join(filepath, filename)
            #     print(file_dir)
            #     file_name = rename_file(file_dir)
            #     file_list.append(file_name)
            if filename.endswith(".txt"):
                file_dir = os.path.join(filepath, filename)
                file_name = rename_file(file_dir)
                print(file_dir)
                file_list.append(file_name)
            elif filename.endswith(".pdf"):
                file_dir = os.path.join(filepath, filename)
                file_name = rename_file(file_dir)
                print(file_dir)
                file_list.append(file_name)
            elif filename.endswith(".docx"):
                file_dir = os.path.join(filepath, filename)
                file_name = rename_file(file_dir)
                print(file_dir)
                file_list.append(file_name)
            # elif "pdf" in filename:
            #         filename1 = filename.split('pdf')[0]+'.pdf'
            #         os.rename(os.path.join(filepath,filename), os.path.join(filepath,filename1))
    
    return file_list

# 加载文件函数
def get_text(dir_path):
        print('dir_path',dir_path)
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
        file_lst = get_files(dir_path)
        if('txt'or'pdf'or'docx'in ''.join(file_lst)):
    # if len(file_lst)>0:
        # docs 存放加载之后的纯文本对象
            docs = []
            # 遍历所有目标文件
            for one_file in tqdm(file_lst):
                print(one_file)
                file_type = one_file.split('.')[-1]
                # if file_type == 'md':
                #     loader = UnstructuredMarkdownLoader(one_file)
                if file_type == 'txt':
                    loader = UnstructuredFileLoader(one_file)
                elif file_type == 'pdf':
                    loader = PyPDFLoader(one_file)
                elif file_type == ('docx'):
                    # loader = DirectoryLoader(one_file,glob="*.doc*", loader_cls=UnstructuredWordDocumentLoader,show_progress=True)
                    loader = Docx2txtLoader(one_file)
                    # loader =  UnstructuredWordDocumentLoader(one_file)
                
                else:
                    # 如果是不符合条件的文件，直接跳过
                    continue
                docs.extend(loader.load())
        return docs

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs[:10])

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 构建向量数据库

# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()

