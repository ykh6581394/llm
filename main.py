# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 07:38:41 2024

@author: 유국현
"""


import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
from openai import OpenAI
import time



st.title('LLM Processor')
#st.set_page_config(layout="wide")
st.sidebar.title("Side Bar")
st.sidebar.checkbox("Checkbox")





#col1,col2 = st.columns(2)
tab1, tab2 = st.tabs(["Write", "Save"])

path_master = st.text_input('Write Path') 

with tab1:
    st.title("Data Loader")
    #txt_file = st.file_uploader("Upload txt",type=['txt'])
    path = path_master
    if st.button("Upload and Classification"):
        
        files = os.listdir(path+"data/")
        master = pd.DataFrame([])
        category_all = []
        for i in range(len(files)):
            df = pd.read_table(path+"data/"+files[i],header=None)
            
            master = pd.concat([master,df])

        st.subheader('Data Summary')
        st.table(master.head())
    

with tab2:
    st.title("Data Saver")
    
    key = st.text_input("Key")
    
    client = OpenAI(
        api_key=key,
    )
    
    
    def gen(x):
        gpt_prompt = [{
            "role"  :  "system",
            "content" : "당신은 친절한 인공지능 챗봇입니다."
            }]
        gpt_prompt.append({
            "role" : "user",
            "content" : x
            })
        gpt_response = client.chat.completions.create(
            messages = gpt_prompt,
            #model = "gpt-3.5-turbo",
            model = "gpt-4-turbo-preview"
            #stream = True
            )
        return gpt_response.choices[0].message.content.strip()
       # return gpt_response["choices"][0]["message"]["content"]

    
    
    pathe = path_master
    if st.button("Category Classification & Save"):
        files = os.listdir(pathe+"data/")
        category_all = []
        file = os.listdir(pathe+"data_label/")
        if len(os.listdir(pathe+"data_label/"))>0:
            for k in range(len(file)):
                os.remove(pathe+"data_label/"+file[k])
        master = pd.DataFrame([])
        mybar = st.progress(0)
        for i in range(len(files)):
            time.sleep(0.01)
            mybar.progress((1+i)*100//len(files))
            df = pd.read_table(pathe+"data/"+files[i],header=None)
            news = str(df[0][0])
            category = gen(news + "이 기사의 카테고리 한 단어만 리턴해")
            category_all.append(category)
            master = pd.concat([master,df])
            shutil.copy(pathe+"data/"+files[i],pathe+"data_label/"+files[i].split(".")[0]+"_"+category+".txt")
        time.sleep(1)
        #mybar.empty()
        master["category"] = category_all
        
        st.subheader('Category Summary')
        st.table(master.head())




