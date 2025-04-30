import streamlit as st
from utils import read_file, show_file_content, load_dotenv_key
from agents import create_vector_store, ask_agent, summarize_content
from charts import generate_chart

st.set_page_config(page_title="IA Analyst", layout="wide")
st.title("📊 IA Analyst - Dashboard com IA")

load_dotenv_key()

uploaded_file = st.file_uploader("📂 Faça upload de um arquivo (PDF, CSV, XLSX)", type=["pdf", "csv", "xlsx"])

if uploaded_file:
    content, docs, df = read_file(uploaded_file)
    st.success("Arquivo carregado com sucesso!")
    show_file_content(content, df)

    vectorstore = create_vector_store(docs)

    tab1, tab2, tab3 = st.tabs(["❓ Perguntas", "🧠 Resumo", "📊 Gráficos"])

    with tab1:
        question = st.text_input("Digite sua pergunta sobre o conteúdo:")
        if question:
            resposta = ask_agent(question, vectorstore)
            st.markdown(f"**Resposta:** {resposta}")

    with tab2:
        if st.button("Gerar resumo"):
            resumo = summarize_content(docs)
            st.markdown(f"### 🧾 Resumo gerado pela IA:\n{resumo}")

    with tab3:
        if df is not None:
            prompt = st.text_input("Descreva o gráfico que deseja gerar (ex: 'Gráfico de barras com vendas por mês'):")
            if st.button("Gerar gráfico") and prompt:
                generate_chart(prompt, df)
        else:
            st.warning("Gráficos disponíveis apenas para arquivos tabulares (CSV/XLSX).")
