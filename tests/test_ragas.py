"""
Тесты Ragas для оценки качества RAG-пайплайна.
Проверяет метрики: Faithfulness, Answer Relevance, Context Recall.
"""
import json
import os
import pytest
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
    from ragas.lls import LangchainLLMWrapper
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
    
    # Сохраняем результаты в JSON
    results_path = Path(__file__).parent / "ragas_results.json"
    results_dict = {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_recall": results["context_recall"],
        "details": eval_data
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    # Проверяем пороги (quality gates)
    assert results["faithfulness"] >= FAITHFULNESS_THRESHOLD, \
        f"Faithfulness {results['faithfulness']:.3f} ниже порога {FAITHFULNESS_THRESHOLD}"
    
    assert results["answer_relevancy"] >= ANSWER_RELEVANCE_THRESHOLD, \
        f"Answer Relevance {results['answer_relevancy']:.3f} ниже порога {ANSWER_RELEVANCE_THRESHOLD}"
    
    assert results["context_recall"] >= CONTEXT_RECALL_THRESHOLD, \
        f"Context Recall {results['context_recall']:.3f} ниже порога {CONTEXT_RECALL_THRESHOLD}"
    
    print(f"\n✅ Ragas результаты:")
    print(f"   Faithfulness: {results['faithfulness']:.3f} (порог: {FAITHFULNESS_THRESHOLD})")
    print(f"   Answer Relevance: {results['answer_relevancy']:.3f} (порог: {ANSWER_RELEVANCE_THRESHOLD})")
    print(f"   Context Recall: {results['context_recall']:.3f} (порог: {CONTEXT_RECALL_THRESHOLD})")


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
    
    # Faithfulness должен быть высоким (ответ не выдуман)
    assert results["faithfulness"] >= 0.8, \
        f"Модель галлюцинирует! Faithfulness: {results['faithfulness']:.3f}"
    
    print(f"\n✅ Тест на галлюцинации пройден. Faithfulness: {results['faithfulness']:.3f}")