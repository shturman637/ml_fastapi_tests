import uvicorn
from fastapi import FastAPI
from transformers import pipeline
import numpy as np
from pydantic import BaseModel

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy

class Item(BaseModel):
    """Данный класс отвечает за определение типа данных, с которым будут работать методы приложения"""
    text: str

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=AutoModelForTokenClassification.from_pretrained(model),tokenizer=AutoTokenizer.from_pretrained(model),*args,**kwargs)

    def postprocess(self, all_outputs):
        """Даннвя функция отвечает за вывод ключевых слов, полученных в результате обрабоки поступившего на вход текста"""
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

app = FastAPI()
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
classifier = pipeline("sentiment-analysis")

@app.post("/predict/")
def predict(item: Item):
    """Метод, посредством которого задействуется извлечение из поступившего на вход текста ключевых слов"""
    return extractor(item.text)[0]

@app.post("/predict2/")
def predict2(item: Item):
    """Метод, посредством которого задействуется определение эмоциональной окраски текста, поступившего на вход"""
    return classifier(item.text)[0]
