"""
Простое RAG-приложение на базе YandexGPT и ChromaDB.
Используется как целевая система для тестирования через Ragas.
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

client = OpenAI(
    api_key=API_KEY,
    project=FOLDER_ID,
    base_url="https://ai.api.cloud.yandex.net/v1"
)


class IMDBRAGPipeline:
    """RAG-пайплайн по отзывам о фильмах."""
    
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self._init_collection()
    
    def _init_collection(self):
        """Инициализирует коллекцию с демо-данными."""
        collection = self.chroma_client.get_or_create_collection(
            name="imdb_reviews",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Демо-данные (в реальном проекте загружаются из датасета)
        documents = [
            "Фильм 'Побег из Шоушенка' — это шедевр кинематографа. Тим Роббинс сыграл Энди Дюфрейна, который был несправедливо осужден за убийство жены. Фильм получил высокие оценки критиков и зрителей.",
            "Кристофер Нолан снял 'Начало' в 2010 году. Главные роли исполнили Леонардо ДиКаприо и Джозеф Гордон-Левитт. Фильм исследует тему сновидений и подсознания.",
            "Сериал 'Во все тяжкие' рассказывает о учителе химии Уолтере Уайте, который начал производить метамфетамин после диагноза рак. Брайан Крэнстон получил Эмми за главную роль.",
            "Фильм 'Интерстеллар' Нолана вышел в 2014 году. Мэттью МакКонахи сыграл астронавта, который путешествует через червоточину в поисках нового дома для человечества.",
            "'Криминальное чтиво' Тарантино 1994 года стало культовым. В фильме снимались Джон Траволта и Сэмюэл Л. Джексон. Фильм получил Золотую пальмовую ветвь.",
            "Сериал 'Игра престолов' выходил с 2011 по 2019 год. Эмилия Кларк сыграла Дейенерис Таргариен. Финальный сезон получил низкие оценки фанатов.",
            "'Бойцовский клуб' 1999 года с Брэдом Питтом и Эдвардом Нортоном стал культовым фильмом. Режиссер — Дэвид Финчер.",
            "Фильм 'Матрица' вышел в 1999 году. Киану Ривз сыграл Нео. Режиссеры — сестры Вачовски. Фильм произвел революцию в экшн-кинематографе.",
            "Сериал 'Друзья' выходил с 1994 по 2004 год. Главные роли исполнили Дженнифер Энистон, Курт Кокс и другие. Сериал стал одним из самых популярных ситкомов.",
            "'Властелин колец' Питера Джексона — эпическая трилогия. Элайджа Вуд сыграл Фродо Бэггинса. Фильм получил 17 Оскаров."
        ]
        
        metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return collection
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Извлекает релевантные документы."""
        results = self.collection.query(query_texts=[query], n_results=k)
        return results["documents"][0]
    
    def generate(self, query: str, contexts: List[str]) -> str:
        """Генерирует ответ на основе контекста."""
        context_text = "\n\n".join(contexts)
        prompt = f"""Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.
Если в контексте нет ответа, скажи "В предоставленных данных нет информации".

Контекст:
{context_text}

Вопрос: {query}

Ответ:"""
        
        response = client.chat.completions.create(
            model=f"gpt://{FOLDER_ID}/yandexgpt",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def run(self, query: str) -> Dict:
        """Полный цикл RAG: retrieval + generation."""
        contexts = self.retrieve(query)
        response = self.generate(query, contexts)
        return {
            "query": query,
            "response": response,
            "contexts": contexts
        }


# Глобальный экземпляр для тестов
pipeline = IMDBRAGPipeline()