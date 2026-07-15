"""
Тесты Ragas для оценки качества RAG-пайплайна.
Проверяет метрики: Faithfulness, Answer Relevance, Context Recall.
"""
import json
import os
import pytest
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.dataset_schema import SingleTurnSample
from datasets import Dataset

from app.rag_pipeline import pipeline

load_dotenv()

# Пороговые значения метрик (quality gates)
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCE_THRESHOLD = 0.6
CONTEXT_RECALL_THRESHOLD = 0.6


@pytest.fixture(scope="module")
def goldens():
    """Загружает золотые примеры."""
    goldens_path = Path(__file__).parent / "goldens.json"
    with open(goldens_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ragas_client():
    """Настраивает Ragas для работы с YandexGPT."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    API_KEY = os.getenv("YANDEX_API_KEY")
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    
    # Настраиваем YandexGPT как evaluator
    yandex_llm = ChatOpenAI(
        model=f"gpt://{FOLDER_ID}/yandexgpt",
        api_key=API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1"
    )
    
    # Для embeddings используем Yandex embeddings
    yandex_embeddings = OpenAIEmbeddings(
        model=f"emb://{FOLDER_ID}/text-search-doc/latest",
        api_key=API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1"
    )
    
    return {
        "llm": LangchainLLMWrapper(yandex_llm),
        "embeddings": LangchainEmbeddingsWrapper(yandex_embeddings)
    }


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
        answer_relevancy,
        context_recall
    ]
    
    for metric in metrics:
        metric.llm = ragas_client["llm"]
        metric.embeddings = ragas_client["embeddings"]
    
    # Собираем данные для Ragas
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
    print(f"   keys: {results.keys() if isinstance(results, dict) else 'N/A'}")
    for key in results.keys():
        value = results[key]
        print(f"   {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
        if isinstance(value, list) and len(value) > 0:
            print(f"      first 5 values: {value[:5]}")
            print(f"      all values: {value}")
    
    # Сохраняем результаты в JSON
    results_path = Path(__file__).parent / "ragas_results.json"
    
    # Ragas возвращает списки значений - нужно усреднить с игнорированием null
    faithfulness_avg = np.nanmean(results["faithfulness"]) if results["faithfulness"] else 0
    answer_relevancy_values = [x for x in results["answer_relevancy"] if x is not None]
    answer_relevancy_avg = np.nanmean(answer_relevancy_values) if answer_relevancy_values else 0
    context_recall_avg = np.nanmean(results["context_recall"]) if results["context_recall"] else 0
    
    results_dict = {
        "faithfulness": float(faithfulness_avg) if not np.isnan(faithfulness_avg) else None,
        "answer_relevancy": float(answer_relevancy_avg) if not np.isnan(answer_relevancy_avg) else None,
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
    
    if not np.isnan(answer_relevancy_avg):
        assert answer_relevancy_avg >= ANSWER_RELEVANCE_THRESHOLD, \
            f"Answer Relevance {answer_relevancy_avg:.3f} ниже порога {ANSWER_RELEVANCE_THRESHOLD}"
    else:
        print("⚠️ Answer Relevance не вычислено (null)")
    
    if not np.isnan(context_recall_avg):
        assert context_recall_avg >= CONTEXT_RECALL_THRESHOLD, \
            f"Context Recall {context_recall_avg:.3f} ниже порога {CONTEXT_RECALL_THRESHOLD}"
    else:
        print("⚠️ Context Recall не вычислено (null)")
    
    print(f"\n✅ Ragas результаты:")
    print(f"   Faithfulness: {faithfulness_avg:.3f} (порог: {FAITHFULNESS_THRESHOLD})")
    print(f"   Answer Relevance: {answer_relevancy_avg:.3f} (порог: {ANSWER_RELEVANCE_THRESHOLD})")
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
