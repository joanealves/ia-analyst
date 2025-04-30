from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import os

SUMMARY_TEMPLATE = """
Você é um assistente analítico. Resuma os principais pontos do texto a seguir de forma clara e objetiva. {custom}
Texto:
{context}
"""

def initialize_openai_components():
    """
    Inicializa componentes do OpenAI com tratamento de erros adequado
    
    Returns:
        Tupla com instâncias de embeddings e llm
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Chave de API OpenAI não encontrada. Verifique seu arquivo .env.")
            st.stop()
            
        embeddings_instance = OpenAIEmbeddings()
        llm_instance = ChatOpenAI(temperature=0)
        return embeddings_instance, llm_instance
    except Exception as e:
        st.error(f"Erro ao inicializar componentes OpenAI: {str(e)}")
        st.error("Verifique sua chave de API OpenAI no arquivo .env.")
        st.stop()

def create_vector_store(docs):
    """
    Cria um armazenamento de vetores a partir dos documentos
    
    Args:
        docs: Documentos para indexação
        
    Returns:
        Vector store FAISS
    """
    embeddings, _ = initialize_openai_components()
    return FAISS.from_documents(docs, embeddings)

def ask_agent(question, vectorstore):
    """
    Pergunta ao agente com base no conteúdo do vectorstore
    
    Args:
        question: Pergunta a ser respondida
        vectorstore: Vector store com os documentos
        
    Returns:
        Resposta do agente
    """
    _, llm = initialize_openai_components()
    
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vectorstore.similarity_search(question)
    resposta = chain.run(input_documents=docs, question=question)
    return resposta

def summarize_content(docs, custom=""):
    """
    Cria um resumo do conteúdo dos documentos
    
    Args:
        docs: Documentos a serem resumidos
        custom: Instruções customizadas para o resumo
        
    Returns:
        Texto do resumo
    """
    _, llm = initialize_openai_components()
    
    prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["context", "custom"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain.run(input_documents=docs, custom=custom)