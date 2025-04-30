import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from utils import read_file, show_file_content
from agents import create_vector_store, ask_agent, summarize_content, initialize_openai_components
from charts import generate_chart
from export_utils import export_to_csv, export_to_excel, export_plotly_to_png, export_summary_to_text, export_analysis_report
from sentiment_analysis import analyze_sentiment, analyze_text_segments
from advanced_search import setup_advanced_retriever, extract_entities, keyword_search, extract_topics

try:
    from predective import train_forecast_model, detect_anomalies
except ImportError:
    st.error("Módulo 'predective' não encontrado. Algumas funcionalidades podem não estar disponíveis.")
    
    def train_forecast_model(df, date_col, value_col, periods=30):
        st.error("Funcionalidade de previsão não disponível!")
        return None, None
        
    def detect_anomalies(df, col, window=5, sigma=3.0):
        st.error("Funcionalidade de detecção de anomalias não disponível!")
        return pd.DataFrame(), None

if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'content' not in st.session_state:
    st.session_state.content = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'advanced_retriever' not in st.session_state:
    st.session_state.advanced_retriever = None
if 'charts' not in st.session_state:
    st.session_state.charts = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""

st.set_page_config(
    page_title="IA Analyst Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar-header {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 10px;
    }
    .results-container {
        margin-top: 20px;
        padding: 15px;
        border-left: 3px solid #1E88E5;
        background-color: #f0f7ff;
    }
    .chart-container {
        margin: 20px 0;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .export-buttons {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def setup_api_key():
    """
    Configura a chave da API OpenAI a partir do arquivo .env
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY não encontrada nas variáveis de ambiente. Por favor adicione em seu arquivo .env.")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key

api_key = setup_api_key()
embeddings, llm = initialize_openai_components()

with st.sidebar:
    st.markdown('<p class="sidebar-header">📁 Dados</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Faça upload de um arquivo", type=["pdf", "csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            content, docs, df = read_file(uploaded_file)
            st.session_state.content = content
            st.session_state.docs = docs
            st.session_state.df = df
            
            if st.session_state.docs:
                st.session_state.vectorstore = create_vector_store(st.session_state.docs)
                st.session_state.advanced_retriever = setup_advanced_retriever(st.session_state.vectorstore, llm)
                st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {str(e)}")
    
    st.markdown('<p class="sidebar-header">⚙️ Configurações</p>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        st.markdown("**Opções de análise numérica:**")
        date_columns = [col for col in st.session_state.df.columns 
                      if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]) 
                      or 'date' in col.lower()]
        
        date_column = st.selectbox("Coluna de data para previsões:", 
                            options=[None] + date_columns,
                            index=0)
        
        numeric_columns = [col for col in st.session_state.df.columns 
                         if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        
        value_column = st.selectbox("Coluna de valor para análise:", 
                           options=[None] + numeric_columns,
                           index=0)

st.markdown('<h1 class="main-header">🤖 IA Analyst Pro</h1>', unsafe_allow_html=True)

if st.session_state.docs is None:
    st.info("👈 Comece fazendo o upload de um arquivo PDF, CSV ou XLSX na barra lateral")
else:
    show_file_content(st.session_state.content, st.session_state.df)
    
    tabs = st.tabs([
        "🔍 Consulta & Resumo", 
        "📊 Visualização", 
        "🔎 Busca Avançada", 
        "😀 Análise de Sentimento",
        "📈 Previsões & Anomalias"
    ])
    
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### ❓ Pergunte sobre seus dados")
            st.markdown("""
            Exemplos de perguntas:
            - Quais são os principais pontos deste documento?
            - Qual foi o total de vendas no último mês?
            - Resuma as informações mais importantes
            """)
            question = st.text_input("Digite sua pergunta:", key="question_input")
            
            if st.button("Buscar resposta", key="answer_btn"):
                if question:
                    with st.spinner("Processando sua pergunta..."):
                        try:
                            if st.session_state.advanced_retriever:
                                docs = st.session_state.advanced_retriever.get_relevant_documents(question)
                                from langchain.chains.question_answering import load_qa_chain
                                chain = load_qa_chain(llm, chain_type="stuff")
                                resposta = chain.run(input_documents=docs, question=question)
                            else:
                                resposta = ask_agent(question, st.session_state.vectorstore)
                            
                            st.markdown('<div class="results-container">', unsafe_allow_html=True)
                            st.markdown(f"**Resposta:**\n\n{resposta}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown(export_summary_to_text(resposta, "resposta_consulta"), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erro ao processar a pergunta: {str(e)}")
        
        with col2:
            st.markdown("### 🧠 Gerar resumo inteligente")
            custom_prompt = st.text_area(
                "Personalize o estilo do resumo (opcional):",
                placeholder="Ex: Foque nos aspectos financeiros. Inclua estatísticas chave.",
                height=100
            )
            
            if st.button("Gerar resumo"):
                with st.spinner("Gerando resumo inteligente..."):
                    try:
                        st.session_state.summary = summarize_content(st.session_state.docs, custom_prompt)
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        st.markdown(f"**Resumo executivo:**\n\n{st.session_state.summary}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown(export_summary_to_text(st.session_state.summary, "resumo_executivo"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro ao gerar resumo: {str(e)}")
    
    with tabs[1]:
        st.markdown("### 📊 Geração de gráficos com IA")
        
        if st.session_state.df is not None:
            st.markdown("""
            Descreva o gráfico que deseja criar em linguagem natural:
            
            Exemplos:
            - Gráfico de barras mostrando vendas por região
            - Gráfico de linha com evolução das vendas ao longo do tempo
            - Gráfico de pizza com a distribuição de produtos por categoria
            """)
            
            chart_prompt = st.text_area("Descrição do gráfico:", height=80)
            
            if st.button("Gerar gráfico"):
                if chart_prompt:
                    with st.spinner("Gerando visualização..."):
                        try:
                            fig = generate_chart(chart_prompt, st.session_state.df)
                            if fig:
                                st.session_state.charts.append(fig)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown('<div class="export-buttons">', unsafe_allow_html=True)
                                st.markdown(export_plotly_to_png(fig, "grafico"), unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.error("Não foi possível gerar o gráfico. Tente uma descrição diferente.")
                        except Exception as e:
                            st.error(f"Erro ao gerar gráfico: {str(e)}")
            
            if len(st.session_state.charts) > 1:
                st.markdown("### Gráficos anteriores")
                for i, fig in enumerate(st.session_state.charts[:-1]):
                    with st.expander(f"Gráfico {i+1}"):
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Para gerar gráficos, faça upload de um arquivo CSV ou XLSX.")
    
    with tabs[2]:
        st.markdown("### 🔎 Busca avançada e extração de entidades")
        
        search_tab1, search_tab2, search_tab3 = st.tabs(["Entidades", "Busca por palavras-chave", "Tópicos principais"])
        
        with search_tab1:
            if st.button("Extrair entidades do texto"):
                with st.spinner("Extraindo entidades..."):
                    try:
                        entities = extract_entities(st.session_state.content, llm)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 👤 Pessoas")
                            if entities["pessoas"]:
                                for pessoa in entities["pessoas"]:
                                    st.markdown(f"- {pessoa}")
                            else:
                                st.info("Nenhuma pessoa identificada")
                                
                            st.markdown("#### 🏢 Organizações")
                            if entities["organizacoes"]:
                                for org in entities["organizacoes"]:
                                    st.markdown(f"- {org}")
                            else:
                                st.info("Nenhuma organização identificada")
                        
                        with col2:
                            st.markdown("#### 📍 Locais")
                            if entities["locais"]:
                                for local in entities["locais"]:
                                    st.markdown(f"- {local}")
                            else:
                                st.info("Nenhum local identificado")
                                
                            st.markdown("#### 📅 Datas")
                            if entities["datas"]:
                                for data in entities["datas"]:
                                    st.markdown(f"- {data}")
                            else:
                                st.info("Nenhuma data identificada")
                        
                        st.markdown("#### 💲 Valores")
                        if entities["valores"]:
                            for valor in entities["valores"]:
                                st.markdown(f"- {valor}")
                        else:
                            st.info("Nenhum valor monetário ou percentual identificado")
                    except Exception as e:
                        st.error(f"Erro na extração de entidades: {str(e)}")
        
        with search_tab2:
            keywords = st.text_input("Digite palavras-chave separadas por vírgula:", placeholder="ex: receita, lucro, investimento")
            context_window = st.slider("Tamanho do contexto (caracteres):", 50, 500, 100)
            
            if st.button("Buscar palavras-chave") and keywords:
                keywords_list = [k.strip() for k in keywords.split(",")]
                with st.spinner("Buscando ocorrências..."):
                    try:
                        results = keyword_search(st.session_state.content, keywords_list, context_window)
                        
                        if results:
                            st.success(f"Encontradas {len(results)} ocorrências")
                            for i, result in enumerate(results):
                                with st.expander(f"Ocorrência {i+1}: '{result['matched_text']}' (Palavra-chave: {result['keyword']})"):
                                    st.markdown("**Contexto:**")
                                    st.markdown(f"...{result['context_before']} **{result['matched_text']}** {result['context_after']}...")
                        else:
                            st.warning("Nenhuma ocorrência encontrada para as palavras-chave informadas.")
                    except Exception as e:
                        st.error(f"Erro na busca por palavras-chave: {str(e)}")
        
        with search_tab3:
            num_topics = st.slider("Número de tópicos a extrair:", 3, 10, 5)
            
            if st.button("Extrair tópicos principais"):
                with st.spinner("Analisando tópicos..."):
                    try:
                        topics, fig = extract_topics(st.session_state.content, llm, num_topics)
                        
                        if topics:
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            for i, topic in enumerate(topics):
                                with st.expander(f"{topic['titulo']} (Relevância: {topic['relevancia']}/10)"):
                                    st.markdown(f"**Descrição:** {topic['descricao']}")
                                    st.markdown(f"**Exemplo no texto:** _{topic['exemplo']}_")
                        else:
                            st.warning("Não foi possível extrair tópicos do texto.")
                    except Exception as e:
                        st.error(f"Erro na extração de tópicos: {str(e)}")
    
    with tabs[3]:
        st.markdown("### 😀 Análise de sentimento")
        
        sentiment_tab1, sentiment_tab2 = st.tabs(["Análise geral", "Análise detalhada"])
        
        with sentiment_tab1:
            if st.button("Analisar sentimento do texto"):
                with st.spinner("Analisando sentimento..."):
                    try:
                        sentiment = analyze_sentiment(st.session_state.content, llm)
                        
                        sentiment_color = {
                            "POSITIVO": "green",
                            "NEUTRO": "gray",
                            "NEGATIVO": "red"
                        }.get(sentiment["sentimento"], "blue")
                        
                        st.markdown(f"<h4 style='color:{sentiment_color}'>Sentimento: {sentiment['sentimento']}</h4>", unsafe_allow_html=True)
                        st.markdown(f"**Pontuação:** {sentiment['pontuacao']:.2f} (de -1 a +1)")
                        st.markdown("**Palavras-chave:**")
                        st.markdown(", ".join(sentiment["palavras_chave"]))
                        st.markdown(f"**Justificativa:** {sentiment['justificativa']}")
                    except Exception as e:
                        st.error(f"Erro na análise de sentimento: {str(e)}")
        
        with sentiment_tab2:
            segment_size = st.slider("Tamanho de cada segmento (caracteres):", 500, 3000, 1000)
            
            if st.button("Realizar análise detalhada de sentimento"):
                with st.spinner("Realizando análise detalhada..."):
                    try:
                        df_results, fig_line, fig_pie, fig_bar = analyze_text_segments(st.session_state.content, llm, segment_size)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_line, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        st.markdown("### Resultados por segmento")
                        st.dataframe(df_results[["segment_index", "sentimento", "pontuacao", "segment_text"]])
                        
                        st.markdown(export_to_csv(df_results, "analise_sentimento"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro na análise detalhada de sentimento: {str(e)}")
    
    with tabs[4]:
        st.markdown("### 📈 Previsões e detecção de anomalias")
        
        if st.session_state.df is not None:
            forecast_tab1, forecast_tab2 = st.tabs(["Previsões", "Detecção de anomalias"])
            
            with forecast_tab1:
                date_cols = [col for col in st.session_state.df.columns if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]) or 'date' in col.lower()]
                numeric_cols = [col for col in st.session_state.df.columns if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
                
                if date_cols and numeric_cols:
                    forecast_date_col = st.selectbox("Selecione a coluna de data:", date_cols)
                    forecast_value_col = st.selectbox("Selecione a coluna de valor a prever:", numeric_cols)
                    forecast_periods = st.slider("Períodos futuros para prever:", 7, 365, 30)
                    
                    if st.button("Gerar previsão"):
                        try:
                            with st.spinner("Gerando modelo de previsão..."):
                                if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[forecast_date_col]):
                                    st.session_state.df[forecast_date_col] = pd.to_datetime(st.session_state.df[forecast_date_col])
                                    
                                forecast_df, forecast_fig = train_forecast_model(
                                    st.session_state.df, 
                                    forecast_date_col, 
                                    forecast_value_col, 
                                    periods=forecast_periods
                                )
                                
                                if forecast_df is not None and forecast_fig is not None:
                                    st.plotly_chart(forecast_fig, use_container_width=True)
                                    
                                    st.markdown("### Valores previstos")
                                    forecast_table = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
                                    forecast_table.columns = ['Data', 'Previsão', 'Limite inferior', 'Limite superior']
                                    st.dataframe(forecast_table)
                                    
                                    st.markdown(export_to_csv(forecast_table, "previsoes"), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erro ao gerar previsão: {str(e)}")
                            st.info("Dica: Verifique se a coluna de data está no formato correto e se há dados suficientes para previsão.")
                else:
                    st.warning("Seu dataset precisa ter pelo menos uma coluna de data e uma coluna numérica para realizar previsões.")
            
            with forecast_tab2:
                if numeric_cols:
                    anomaly_col = st.selectbox("Selecione a coluna para detecção de anomalias:", numeric_cols)
                    window_size = st.slider("Tamanho da janela para análise:", 3, 20, 5)
                    sigma_threshold = st.slider("Limiar de desvio padrão (σ):", 1.0, 5.0, 3.0, 0.1)
                    
                    if st.button("Detectar anomalias"):
                        try:
                            with st.spinner("Analisando anomalias..."):
                                df_analysis = st.session_state.df.copy()
                                
                                anomalies_df, anomalies_fig = detect_anomalies(
                                    df_analysis, 
                                    anomaly_col, 
                                    window=window_size, 
                                    sigma=sigma_threshold
                                )
                                
                                if anomalies_fig is not None:
                                    st.plotly_chart(anomalies_fig, use_container_width=True)
                                
                                if anomalies_df is not None and not anomalies_df.empty:
                                    st.markdown(f"### Anomalias detectadas: {len(anomalies_df)}")
                                    st.dataframe(anomalies_df)
                                    
                                    st.markdown(export_to_csv(anomalies_df, "anomalias_detectadas"), unsafe_allow_html=True)
                                else:
                                    st.success("Nenhuma anomalia detectada com os parâmetros especificados.")
                        except Exception as e:
                            st.error(f"Erro na detecção de anomalias: {str(e)}")
                else:
                    st.warning("Seu dataset precisa ter pelo menos uma coluna numérica para detecção de anomalias.")
        else:
            st.info("Para gerar previsões e detectar anomalias, faça upload de um arquivo CSV ou XLSX.")

    if st.session_state.summary and len(st.session_state.charts) > 0:
        st.markdown("---")
        st.markdown("### 📑 Exportar relatório completo")
        if st.button("Gerar relatório de análise"):
            try:
                report_link = export_analysis_report(
                    st.session_state.content if st.session_state.df is None else st.session_state.df,
                    st.session_state.charts,
                    st.session_state.summary,
                    "relatorio_ia_analyst"
                )
                st.markdown(report_link, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {str(e)}")