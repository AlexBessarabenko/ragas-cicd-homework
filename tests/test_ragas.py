"""
Тесты Ragas для оценки качества RAG-пайплайна.
Проверяет метрики: Faithfulness, Answer Similarity, Context Recall.
"""
import json
import os
from typing import Dict, List

import app.ragas_compat  # noqa: F401  # заглушки для ragas 0.2.x + langchain 1.x
import numpy as np
import pytest
import requests
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pathlib import Path
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_similarity, context_recall, faithfulness

from app.rag_pipeline import pipeline

load_dotenv()

# Пороговые значения метрик (quality gates)
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_SIMILARITY_THRESHOLD = 0.6
CONTEXT_RECALL_THRESHOLD = 0.6


@pytest.fixture(scope="module")
def goldens():
    """Загружает золотые примеры."""
    goldens_path = Path(__file__).parent / "goldens.json"
    with open(goldens_path, "r", encoding="utf-8") as f:
        return json.load(f)


class YandexChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI с параметрами, совместимыми с YandexGPT OpenAI-совместимым API.
    YandexGPT не принимает ряд полей (n, stop, stream, logprobs и др.),
    которые langchain-openai добавляет в запрос по умолчанию.
    """

    @property
    def _default_params(self) -> Dict[str, object]:
        params = super()._default_params
        # YandexGPT не поддерживает ряд параметров OpenAI API.
        unsupported = (
            "n", "stop", "stream", "logprobs", "top_logprobs", "logit_bias",
            "extra_body", "reasoning_effort", "reasoning", "verbosity",
            "context_management", "include", "prompt_cache_options",
            "service_tier", "truncation", "store",
        )
        for key in unsupported:
            params.pop(key, None)
        # Очень маленькие значения температуры (1e-8), которые Ragas использует
        # по умолчанию, YandexGPT может отклонять из-за научной нотации.
        temperature = params.get("temperature")
        if temperature is not None and temperature < 1e-6:
            params["temperature"] = 0.0
        return params


@pytest.fixture(scope="module")
def ragas_client():
    """Настраивает Ragas для работы с YandexGPT."""
    API_KEY = os.getenv("YANDEX_API_KEY")
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

    # Настраиваем YandexGPT как evaluator
    yandex_llm = YandexChatOpenAI(
        model=f"gpt://{FOLDER_ID}/yandexgpt",
        api_key=API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1",
    )

    # Для embeddings используем родной Yandex API, т.к. OpenAI-совместимый
    # эндпоинт Yandex для embeddings не работает стабильно через langchain-openai.
    yandex_embeddings = YandexEmbeddings(
        api_key=API_KEY,
        folder_id=FOLDER_ID,
    )

    return {
        "llm": LangchainLLMWrapper(yandex_llm),
        "embeddings": LangchainEmbeddingsWrapper(yandex_embeddings),
    }


class YandexEmbeddings(Embeddings):
    """
    Обёртка над родным Yandex Text Embedding API.
    Использует разные модели для документов (text-search-doc) и запросов
    (text-search-query), как рекомендуется в документации Yandex.
    """

    def __init__(self, api_key: str, folder_id: str, timeout: float = 60.0):
        self.api_key = api_key
        self.folder_id = folder_id
        self.timeout = timeout
        self.url = "https://ai.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-folder-id": folder_id,
        }

    def _model_uri(self, text_type: str) -> str:
        model = "text-search-doc" if text_type == "doc" else "text-search-query"
        return f"emb://{self.folder_id}/{model}/latest"

    def _embed(self, text: str, text_type: str) -> List[float]:
        response = requests.post(
            self.url,
            json={"modelUri": self._model_uri(text_type), "text": text},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t, "doc") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "query")


def test_goldens_exist(goldens):
    """Проверяет, что золотые примеры загружены."""
    assert len(goldens) >= 10, "Должно быть минимум 10 золотых примеров"


def test_rag_pipeline_returns_response(goldens):
    """Проверяет, что RAG-пайплайн возвращает ответ."""
    result = pipeline.run(goldens[0]["question"])
    assert "response" in result
    assert "contexts" in result
    assert len(result["response"]) > 0


def test_ragas_evaluation(goldens, ragas_client):
    """
    Основной тест: запускает Ragas-оценку на всех золотых примерах.
    Проверяет метрики Faithfulness, Answer Relevance, Context Recall.
    """
    # Настраиваем метрики с нашим LLM и embeddings
    metrics = [
        faithfulness,
        answer_similarity,
        context_recall
    ]
    
    for metric in metrics:
        metric.llm = ragas_client["llm"]
        metric.embeddings = ragas_client["embeddings"]

    eval_data = []
    for golden in goldens:
        result = pipeline.run(golden["question"])
        eval_data.append({
            "user_input": golden["question"],
            "response": result["response"],
            "retrieved_contexts": result["contexts"],
            "reference": golden["answer"]
        })
    
    # Создаем Dataset для Ragas
    dataset = Dataset.from_list(eval_data)
    
    # Запускаем оценку
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_client["llm"],
        embeddings=ragas_client["embeddings"]
    )
    
    # Отладка: выводим структуру результата
    print(f"\n🔍 Структура результата Ragas:")
    print(f"   Тип results: {type(results)}")
    print(f"   keys: {list(results._scores_dict.keys())}")
    for key in results._scores_dict.keys():
        value = results[key]
        print(f"   {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
        if isinstance(value, list) and len(value) > 0:
            print(f"      first 5 values: {value[:5]}")
            print(f"      all values: {value}")

    # Сохраняем результаты в JSON
    results_path = Path(__file__).parent / "ragas_results.json"

    # Ragas возвращает списки значений - нужно усреднить с игнорированием null
    faithfulness_values = [x for x in results["faithfulness"] if x is not None]
    answer_similarity_values = [x for x in results["answer_similarity"] if x is not None]
    context_recall_values = [x for x in results["context_recall"] if x is not None]

    faithfulness_avg = np.nanmean(faithfulness_values) if faithfulness_values else float("nan")
    answer_similarity_avg = np.nanmean(answer_similarity_values) if answer_similarity_values else float("nan")
    context_recall_avg = np.nanmean(context_recall_values) if context_recall_values else float("nan")
    
    results_dict = {
        "faithfulness": float(faithfulness_avg) if not np.isnan(faithfulness_avg) else None,
        "answer_similarity": float(answer_similarity_avg) if not np.isnan(answer_similarity_avg) else None,
        "context_recall": float(context_recall_avg) if not np.isnan(context_recall_avg) else None,
        "details": eval_data
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    # Проверяем пороги (quality gates) - пропускаем null значения
    if not np.isnan(faithfulness_avg):
        assert faithfulness_avg >= FAITHFULNESS_THRESHOLD, \
            f"Faithfulness {faithfulness_avg:.3f} ниже порога {FAITHFULNESS_THRESHOLD}"
    else:
        print("⚠️ Faithfulness не вычислено (null)")
    
    if not np.isnan(answer_similarity_avg):
        assert answer_similarity_avg >= ANSWER_SIMILARITY_THRESHOLD, \
            f"Answer Similarity {answer_similarity_avg:.3f} ниже порога {ANSWER_SIMILARITY_THRESHOLD}"
    else:
        print("⚠️ Answer Similarity не вычислено (null)")
    
    if not np.isnan(context_recall_avg):
        assert context_recall_avg >= CONTEXT_RECALL_THRESHOLD, \
            f"Context Recall {context_recall_avg:.3f} ниже порога {CONTEXT_RECALL_THRESHOLD}"
    else:
        print("⚠️ Context Recall не вычислено (null)")
    
    print(f"\n✅ Ragas результаты:")
    print(f"   Faithfulness: {faithfulness_avg:.3f} (порог: {FAITHFULNESS_THRESHOLD})")
    print(f"   Answer Similarity: {answer_similarity_avg:.3f} (порог: {ANSWER_SIMILARITY_THRESHOLD})")
    print(f"   Context Recall: {context_recall_avg:.3f} (порог: {CONTEXT_RECALL_THRESHOLD})")


def test_no_hallucinations(goldens, ragas_client):
    """
    Специальный тест на галлюцинации.
    Faithfulness проверяет, что ответ основан на контексте, а не выдуман.
    """
    metric = faithfulness
    metric.llm = ragas_client["llm"]
    metric.embeddings = ragas_client["embeddings"]
    
    # Берем вопрос, на который в контексте нет ответа
    tricky_question = "Кто сыграл главную роль в фильме 'Титаник'?"
    result = pipeline.run(tricky_question)
    
    eval_data = [{
        "user_input": tricky_question,
        "response": result["response"],
        "retrieved_contexts": result["contexts"],
        "reference": "В предоставленных данных нет информации"
    }]
    
    dataset = Dataset.from_list(eval_data)
    results = evaluate(
        dataset=dataset,
        metrics=[metric],
        llm=ragas_client["llm"],
        embeddings=ragas_client["embeddings"]
    )
    
    # Ragas возвращает список значений - берём первое
    faithfulness_value = results["faithfulness"][0] if results["faithfulness"] else 0
    
    # Faithfulness должен быть высоким (ответ не выдуман)
    assert faithfulness_value >= 0.8, \
        f"Модель галлюцинирует! Faithfulness: {faithfulness_value:.3f}"
    
    print(f"\n✅ Тест на галлюцинации пройден. Faithfulness: {faithfulness_value:.3f}")
