import plotly.express as px
import pandas as pd
import os
from openai import OpenAI

def generate_chart(prompt, df):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        columns = ", ".join(df.columns)
        question = f"Crie um código Python usando Plotly Express para gerar o gráfico a partir dos dados com as colunas: {columns}. O gráfico deve atender ao seguinte pedido: {prompt}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em visualização de dados com Plotly."},
                {"role": "user", "content": question}
            ]
        )

        code = response.choices[0].message.content
        local_vars = {"px": px, "pd": pd, "df": df}
        exec(code, {}, local_vars)
        fig = local_vars.get("fig", None)
        return fig

    except Exception as e:
        print("Erro ao gerar gráfico:", e)
        return None