from langchain.schema import Document
from io import BytesIO
import pandas as pd
import fitz  
import streamlit as st
import os
from dotenv import load_dotenv

def read_file(uploaded_file):
    docs = []
    df = None
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        with BytesIO(uploaded_file.read()) as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        docs = [Document(page_content=text)]
        return text, docs, None

    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        text = df.to_csv(index=False)
        docs = [Document(page_content=text)]
        return text, docs, df

    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        text = df.to_csv(index=False)
        docs = [Document(page_content=text)]
        return text, docs, df

    else:
        return "Tipo de arquivo não suportado", [], None

def show_file_content(text, df):
    st.subheader("📄 Prévia do conteúdo")
    if df is not None:
        st.dataframe(df.head())
    else:
        st.text_area("Conteúdo do arquivo:", text[:3000], height=300)

def load_dotenv_key():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
