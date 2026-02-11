import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from typing import List, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, "..", "models")


def model_load(path: str) -> Tuple[BertForSequenceClassification, AutoTokenizer]:
    """Загружает предобученную модель и токенизатор.

    Args:
        path: Путь к директории, содержащей файлы модели (config.json, weights.bin)
        и файлы токенизатора.

    Returns:
        Кортеж содержащий (модель, токенизатор).
    """
    model = BertForSequenceClassification.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.to(device)

    return model, tokenizer


model, tokenizer = model_load(PATH)
model.eval()


def inference(
    model: BertForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    messages: List[str]
) -> List[int]:
    """Функция предсказания текста на наличие в нем спама.

    Args:
        model: Загруженная обученная модель.
        tokenizer: Загруженный токенизатор для данной модели.
        messages: Текст(ы) для инференса, в виде списка.

    Returns:
        Список содержащий метки предсказанных классов [1: спам, 0: не спам]
    """

    inputs = tokenizer(
        messages,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs.to(device))
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()

    return predictions
