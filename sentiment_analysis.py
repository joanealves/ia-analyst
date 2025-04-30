from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import plotly.express as px
import pandas as pd
import json
import re
from collections import Counter

def analyze_sentiment(text, llm):
    """
    Analisa o sentimento de um texto usando LLM
    
    Args:
        text: Texto a ser analisado
        llm: Modelo de linguagem (ChatOpenAI)
        
    Returns:
        Dicionário com sentimento, pontuação e justificativa
    """
    template = """
    Analise o sentimento do texto a seguir e classifique como POSITIVO, NEGATIVO ou NEUTRO.
    Forneça uma pontuação de -1 (muito negativo) a +1 (muito positivo).
    Identifique até 5 palavras-chave que justifiquem sua classificação.
    
    Texto para análise:
    {text}
    
    Responda no formato JSON exato abaixo:
    ```json
    {
        "sentimento": "POSITIVO|NEGATIVO|NEUTRO",
        "pontuacao": 0.0,
        "palavras_chave": ["palavra1", "palavra2", "..."],
        "justificativa": "Breve justificativa da classificação"
    }
    ```
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    sentiment_chain = LLMChain(llm=llm, prompt=prompt)
    
    text_sample = text[:5000]  
    
    response = sentiment_chain.run(text=text_sample)
    
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response
    
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        result = {
            "sentimento": "NEUTRO",
            "pontuacao": 0.0,
            "palavras_chave": [],
            "justificativa": "Não foi possível analisar o sentimento."
        }
    
    return result

def analyze_text_segments(text, llm, segment_size=1000, overlap=200):
    """
    Divide o texto em segmentos e analisa o sentimento de cada segmento
    
    Args:
        text: Texto completo a ser analisado
        llm: Modelo de linguagem
        segment_size: Tamanho de cada segmento em caracteres
        overlap: Sobreposição entre segmentos consecutivos
        
    Returns:
        DataFrame com os resultados e figura Plotly
    """
    segments = []
    
    paragraphs = text.split('\n')
    
    current_segment = ""
    segment_index = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if len(current_segment) + len(paragraph) > segment_size:
            segments.append((segment_index, current_segment))
            segment_index += 1
            
            words = current_segment.split()
            overlap_text = " ".join(words[-int(overlap/10):]) if words else ""
            current_segment = overlap_text + " " + paragraph
        else:
            current_segment += " " + paragraph
    
    if current_segment:
        segments.append((segment_index, current_segment))
    
    results = []
    
    for idx, segment_text in segments:
        result = analyze_sentiment(segment_text, llm)
        result["segment_index"] = idx
        result["segment_text"] = segment_text[:100] + "..." if len(segment_text) > 100 else segment_text
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    fig_line = px.line(
        df_results, 
        x="segment_index", 
        y="pontuacao", 
        title="Evolução do Sentimento ao Longo do Texto",
        labels={"segment_index": "Segmento", "pontuacao": "Pontuação de Sentimento"},
        markers=True
    )
    
    fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
    
    sentiment_counts = Counter(df_results["sentimento"])
    
    df_pie = pd.DataFrame({
        "Sentimento": list(sentiment_counts.keys()),
        "Contagem": list(sentiment_counts.values())
    })
    
    fig_pie = px.pie(
        df_pie, 
        values="Contagem", 
        names="Sentimento", 
        title="Distribuição de Sentimentos",
        color="Sentimento",
        color_discrete_map={
            "POSITIVO": "green",
            "NEUTRO": "gray",
            "NEGATIVO": "red"
        }
    )
    
    all_keywords = []
    for keywords in df_results["palavras_chave"]:
        all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(10)
    
    df_keywords = pd.DataFrame({
        "Palavra": [kw[0] for kw in top_keywords],
        "Frequência": [kw[1] for kw in top_keywords]
    })
    
    fig_bar = px.bar(
        df_keywords,
        x="Palavra",
        y="Frequência",
        title="Principais Palavras-Chave",
        color="Frequência",
        color_continuous_scale="Viridis"
    )
    
    return df_results, fig_line, fig_pie, fig_bar