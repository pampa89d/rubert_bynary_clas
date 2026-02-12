import streamlit as st
import pandas as pd
import requests

URL = "http://spam-api:8000/spam_classification/"

st.title("Сервис для детекции спама в текстовых сообщениях")

texts = st.text_area(
    label="Сообщения для детекции спама",
    help="Каждое новое сообщение начинается с переноса строки",
)
messages = list(texts.split("\n"))

data = {"messages": messages}

response = requests.post(URL, json=data)

if response.status_code == 200:
    df = pd.DataFrame(
        {
            "messages": data["messages"],
            "prediction_label": response.json()["predictions"],
        }
    )
    st.write(df)
else:
    st.write(f"Ошибка {response.status_code}: {response.text}")
