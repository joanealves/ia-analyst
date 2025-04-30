import base64
import io
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.io as pio
import json
import os
from datetime import datetime

def export_to_csv(df, filename="dados_exportados"):
    """
    Cria um link para download do DataFrame como arquivo CSV
    
    Args:
        df: DataFrame pandas a ser exportado
        filename: Nome do arquivo (sem extensão)
    
    Returns:
        Link HTML para download
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}_{timestamp}.csv">Download CSV</a>'
    return href

def export_to_excel(df, filename="dados_exportados"):
    """
    Cria um link para download do DataFrame como arquivo Excel
    
    Args:
        df: DataFrame pandas a ser exportado
        filename: Nome do arquivo (sem extensão)
    
    Returns:
        Link HTML para download
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}_{timestamp}.xlsx">Download Excel</a>'
    return href

def export_plotly_to_png(fig, filename="grafico"):
    """
    Cria um link para download de um gráfico Plotly como imagem PNG
    
    Args:
        fig: Figura Plotly a ser exportada
        filename: Nome do arquivo (sem extensão)
    
    Returns:
        Link HTML para download
    """
    img_bytes = pio.to_image(fig, format="png")
    
    b64 = base64.b64encode(img_bytes).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}_{timestamp}.png">Download PNG</a>'
    return href

def export_summary_to_text(summary, filename="resumo"):
    """
    Cria um link para download de um resumo como arquivo de texto
    
    Args:
        summary: Texto do resumo
        filename: Nome do arquivo (sem extensão)
    
    Returns:
        Link HTML para download
    """
    b64 = base64.b64encode(summary.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}_{timestamp}.txt">Download Texto</a>'
    return href

def export_analysis_report(content, charts, summary, filename="relatorio_analise"):
    """
    Cria um arquivo HTML com relatório completo de análise e um link para download
    
    Args:
        content: Texto ou DataFrame do conteúdo original
        charts: Lista de figuras Plotly
        summary: Texto do resumo gerado
        filename: Nome do arquivo (sem extensão)
    
    Returns:
        Link HTML para download
    """
    chart_html = ""
    for i, fig in enumerate(charts):
        chart_html += f"<div class='chart'><h3>Gráfico {i+1}</h3>"
        chart_html += pio.to_html(fig, full_html=False)
        chart_html += "</div>"
    
    if isinstance(content, pd.DataFrame):
        data_html = f"<h2>Dados Originais</h2><div class='data-table'>{content.head(10).to_html()}</div>"
    else:
        data_html = f"<h2>Conteúdo Original</h2><div class='data-content'><pre>{content[:3000]}</pre></div>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Análise - IA Analyst</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
            .summary {{ background-color: #f2f2f2; padding: 15px; margin: 20px 0; border-left: 5px solid #4CAF50; }}
            .chart {{ margin: 30px 0; }}
            .data-table {{ margin: 20px 0; max-height: 400px; overflow: auto; }}
            .data-content {{ margin: 20px 0; max-height: 300px; overflow: auto; }}
            pre {{ white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Relatório de Análise - IA Analyst</h1>
            <p>Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
        </div>
        
        <h2>Resumo Executivo</h2>
        <div class="summary">
            {summary}
        </div>
        
        {data_html}
        
        <h2>Visualizações</h2>
        {chart_html}
        
        <footer>
            <p>Gerado por IA Analyst - Relatório automatizado</p>
        </footer>
    </body>
    </html>
    """
    
    b64 = base64.b64encode(html.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}_{timestamp}.html">Download Relatório Completo (HTML)</a>'
    return href