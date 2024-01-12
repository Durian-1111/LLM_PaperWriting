#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   llm.py
@Time    :   2023/10/16 18:53:26
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于InternLM模型自定义 LLM 类
'''

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
import torch

class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,device_map = 'auto')#load_in_4bit=True
        # tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        # tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        # print("Creat model...")
        # model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16, trust_remote_code=True)#torch_dtype=torch.bfloat16, 

        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """你现在是数学老师，仔细审题，详细解题答题 """ 
        #"""You are an AI assistant whose name is InternLM (书生·浦语).
        #               - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        #                - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        #                """
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"
    
if __name__ == "__main__":
    # 测试代码
    llm = InternLM_LLM(model_path = "/home/xlab-app-center/InternLM-chat-7b-8k")
    print(llm._call("你是谁"))

