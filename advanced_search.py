from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import plotly.express as px
import re
import json

def setup_advanced_retriever(vectorstore, llm):
    """
    Configura um retriever avançado com compressão contextual
    
    Args:
        vectorstore: O vectorstore FAISS
        llm: O modelo de linguagem
        
    Returns:
        O retriever configurado
    """
    prompt_template = """
    Usuário está buscando a resposta para: {query}
    
    Temos o seguinte texto extraído que pode conter a resposta:
    {context}
    
    Identifique e extraia apenas as partes específicas que respondem diretamente a questão, ignorando informações irrelevantes.
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"]
    )
    
    compressor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    return compression_retriever

def extract_entities(text, llm):
    """
    Extrai entidades nomeadas (pessoas, organizações, locais, datas) do texto
    
    Args:
        text: O texto a ser analisado
        llm: O modelo de linguagem
        
    Returns:
        Dicionário com as entidades extraídas
    """
    template = """
    Extraia as entidades nomeadas do texto a seguir. Identifique:
    - Pessoas (nomes completos)
    - Organizações (empresas, instituições)
    - Locais (cidades, países, regiões)
    - Datas e períodos temporais
    - Valores monetários e percentuais
    
    Texto:
    {text}
    
    Forneça a resposta como JSON no formato exato abaixo:
    ```json
    {
        "pessoas": ["nome1", "nome2", ...],
        "organizacoes": ["org1", "org2", ...],
        "locais": ["local1", "local2", ...],
        "datas": ["data1", "data2", ...],
        "valores": ["valor1", "valor2", ...]
    }
    ```
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    text_sample = text[:4000]
    
    response = chain.run(text=text_sample)
    
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response
    
    try:
        entities = json.loads(json_str)
    except json.JSONDecodeError:
        entities = {
            "pessoas": [],
            "organizacoes": [],
            "locais": [],
            "datas": [],
            "valores": []
        }
    
    return entities

def keyword_search(text, keywords, context_window=100):
    """
    Realiza busca por palavras-chave no texto com contexto
    
    Args:
        text: O texto onde buscar
        keywords: Lista de palavras-chave ou frases
        context_window: Tamanho do contexto antes/depois em caracteres
        
    Returns:
        Lista de ocorrências com contexto
    """
    results = []
    
    for keyword in keywords:
        escaped_keyword = re.escape(keyword)
        for match in re.finditer(escaped_keyword, text, re.IGNORECASE):
            start, end = match.span()
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            
            before = text[context_start:start]
            matched = text[start:end]
            after = text[end:context_end]
            
            results.append({
                "keyword": keyword,
                "matched_text": matched,
                "context_before": before,
                "context_after": after,
                "position": start,
                "full_context": f"{before}[{matched}]{after}"
            })
    
    results.sort(key=lambda x: x["position"])
    
    return results

def extract_topics(text, llm, num_topics=5):
    """
    Extrai os principais tópicos do texto
    
    Args:
        text: O texto a ser analisado
        llm: O modelo de linguagem
        num_topics: Número de tópicos a extrair
        
    Returns:
        Lista de tópicos com pontuação e exemplos
    """
    template = """
    Analise o texto a seguir e extraia os {num_topics} principais tópicos abordados.
    Para cada tópico, forneça:
    - Um título conciso para o tópico
    - Uma breve descrição (1-2 frases)
    - Uma pontuação de relevância de 1 a 10
    - Uma frase de exemplo extraída do texto
    
    Texto:
    {text}
    
    Responda no formato JSON exato abaixo:
    ```json
    [
        {
            "titulo": "Título do tópico 1",
            "descricao": "Breve descrição do tópico",
            "relevancia": 9,
            "exemplo": "Frase extraída do texto original"
        },
        ...
    ]
    ```
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text", "num_topics"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    text_sample = text[:5000]
    
    response = chain.run(text=text_sample, num_topics=num_topics)
    
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response
    
    try:
        topics = json.loads(json_str)
    except json.JSONDecodeError:
        topics = []
    
    df_topics = pd.DataFrame(topics)
    if not df_topics.empty and "relevancia" in df_topics.columns and "titulo" in df_topics.columns:
        fig = px.bar(
            df_topics, 
            x="titulo", 
            y="relevancia", 
            title="Principais Tópicos por Relevância",
            labels={"titulo": "Tópico", "relevancia": "Relevância (1-10)"},
            color="relevancia",
            text="relevancia"
        )
        
        fig.update_layout(xaxis_tickangle=-45)
    else:
        fig = None
    
    return topics, fig