from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List
import inference_model

app = FastAPI(title="Spam Detector API")


class InputData(BaseModel):
    messages: List[str]


class OutputData(BaseModel):
    predictions: List[int]


model = inference_model.model
tokenizer = inference_model.tokenizer


@app.post("/spam_classification/", response_model=OutputData)
def predictions(data: InputData):
    """
    Классификация списка текстов на наличие спама с использованием модели BERT.
    
    Args:
        messages: Список строк (текстов сообщений), в формате ["msg1", "msg2", ...]

    Returns:
        return: Список меток (1 — спам, 0 — не спам), в формате {"predictions": int}

    Использует предобученную модель `rubert-tiny2` для высокоскоростного инференса.
    """
    if not data.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Передан пустой список текстов.",
        )

    result = inference_model.inference(model, tokenizer, data.messages)
    return {"predictions": result}
