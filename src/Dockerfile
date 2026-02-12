FROM python:3.10-slim
# apt-get update - скачивает актуальные индексы пакетов из репозиториев Ubuntu
# apt-get install -y build-essential - включает в себя компиляторы gcc, g++ и утилиту make
# rm -rf /var/lib/apt/lists/* - удаление скачанных временных файлов
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Папку models будем подключать отдельно через volumes
COPY . ./src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
