import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

def train_forecast_model(df, date_column, value_column, periods=30):
    """
    Treina um modelo Prophet para previsão de séries temporais
    
    Args:
        df: DataFrame contendo os dados
        date_column: Nome da coluna de data
        value_column: Nome da coluna de valor a ser previsto
        periods: Número de períodos futuros para prever
        
    Returns:
        DataFrame com previsões e figura Plotly
    """
    prophet_df = df[[date_column, value_column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    model = Prophet(interval_width=0.95)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='markers+lines',
        name='Dados históricos',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Previsão',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Intervalo de confiança 95%'
    ))
    
    fig.update_layout(
        title=f'Previsão para {value_column} nos próximos {periods} períodos',
        xaxis_title='Data',
        yaxis_title=value_column,
        legend_title='Legenda',
        template='plotly_white'
    )
    
    return forecast, fig

def detect_anomalies(df, value_column, window=5, sigma=3):
    """
    Detecta anomalias em uma série de dados usando o método de desvio padrão
    
    Args:
        df: DataFrame com os dados
        value_column: Nome da coluna a ser analisada
        window: Tamanho da janela móvel para calcular média e desvio
        sigma: Número de desvios padrão para considerar como anomalia
        
    Returns:
        DataFrame com anomalias marcadas e figura Plotly
    """
    rolling_mean = df[value_column].rolling(window=window).mean()
    rolling_std = df[value_column].rolling(window=window).std()
    
    upper_limit = rolling_mean + (rolling_std * sigma)
    lower_limit = rolling_mean - (rolling_std * sigma)
    
    anomalies = df[(df[value_column] > upper_limit) | (df[value_column] < lower_limit)].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[value_column],
        mode='lines',
        name='Valores',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=upper_limit,
        mode='lines',
        name='Limite Superior',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=lower_limit,
        mode='lines',
        name='Limite Inferior',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies[value_column],
        mode='markers',
        name='Anomalias',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f'Detecção de Anomalias para {value_column}',
        xaxis_title='Índice',
        yaxis_title=value_column,
        legend_title='Legenda',
        template='plotly_white'
    )
    
    return anomalies, fig