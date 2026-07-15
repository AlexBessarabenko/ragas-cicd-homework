# AGENTS.md

Файл для AI-ассистентов, работающих с проектом. Проект небольшой, поэтому ниже собраны факты, которые нужны для понимания архитектуры, сборки, тестирования и безопасности.

## Обзор проекта

Репозиторий демонстрирует CI/CD-пайплайн для тестирования LLM-приложения на базе RAG с помощью фреймворка **Ragas**. Целевая система — простой RAG-пайплайн над отзывами о фильмах:

- **ChromaDB** используется как векторная база данных (in-memory, демо-данные).
- **YandexGPT** выступает генеративной моделью.
- **Ragas** оценивает качество ответов по трём метрикам: `Faithfulness`, `Answer Relevancy`, `Context Recall`.

Пайплайн падает, если качество ответов ниже заданных порогов (quality gates).

## Технологический стек

- **Язык:** Python 3.14 (задано в CI).
- **Основные зависимости:**
  - `ragas==0.2.14`
  - `langchain==1.3.13`, `langchain-core==1.4.9`, `langchain-openai==1.3.5`, `langchain-community==0.4.2`
  - `chromadb==1.5.9`
  - `openai==2.45.0` (используется для совместимости с API YandexGPT)
  - `datasets==5.0.0`, `pandas==3.0.3`
  - `pytest==9.1.1`, `pytest-html==4.2.0`
  - `python-dotenv==1.2.2`
- **Конфигурация зависимостей:** обычный `requirements.txt`. Никаких `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`, `pytest.ini` в репозитории нет.


## Структура проекта

```
.
├── app/
│   ├── __init__.py          # Пустая инициализация пакета
│   └── rag_pipeline.py      # RAG-пайплайн (IMDBRAGPipeline)
├── tests/
│   ├── goldens.json         # 10 золотых примеров (question / answer / contexts)
│   └── test_ragas.py        # Тесты pytest + Ragas
├── .github/workflows/
│   └── ci.yml               # GitHub Actions workflow
├── requirements.txt
├── .env.example             # Шаблон переменных окружения
├── .env                     # Локальные секреты (в .gitignore)
└── README.md
```

## Архитектура и организация кода

### `app/rag_pipeline.py`

- Класс `IMDBRAGPipeline` — весь RAG-цикл.
- При создании экземпляра инициализируется `chromadb.Client()` и коллекция `imdb_reviews` с метрикой расстояния `cosine`.
- В коллекцию жёстко зашиты 10 демонстрационных документов об отзывах/фильмах.
- `retrieve(query, k=3)` — возвращает `k` ближайших документов.
- `generate(query, contexts)` — формирует промпт с инструкцией отвечать только по контексту и вызывает YandexGPT через OpenAI-совместимый клиент.
- `run(query)` — объединяет retrieval и generation, возвращает словарь `{query, response, contexts}`.
- В конце модуля создаётся глобальный экземпляр `pipeline = IMDBRAGPipeline()`, который используется в тестах.

### `tests/test_ragas.py`

- `goldens` (fixture, scope `module`) — загружает `tests/goldens.json`.
- `ragas_client` (fixture, scope `module`) — настраивает YandexGPT как evaluator LLM и Yandex embeddings для Ragas. Для LLM используется кастомный `YandexChatOpenAI`, который убирает из запроса поля `n`, `stop`, `stream`, `logprobs`, `reasoning` и др., неподдерживаемые YandexGPT OpenAI-совместимым API.
- Тесты:
  1. `test_goldens_exist` — проверяет, что в `goldens.json` не менее 10 примеров.
  2. `test_rag_pipeline_returns_response` — дымовой тест, что пайплайн возвращает непустой ответ.
  3. `test_ragas_evaluation` — основная оценка по всем золотым примерам, сохраняет `tests/ragas_results.json` и проверяет quality gates.
  4. `test_no_hallucinations` — проверяет вопрос без ответа в контексте (`"Кто сыграл главную роль в фильме 'Титаник'?"`), требует `Faithfulness >= 0.8`.

### Пороговые значения качества

| Метрика | Порог в тестах | Порог в CI |
|---------|----------------|------------|
| Faithfulness | `>= 0.7` | `>= 0.7` |
| Answer Relevancy | `>= 0.6` | `>= 0.6` |
| Context Recall | `>= 0.6` | `>= 0.6` |

Если метрика не вычислена (`null` / `NaN`), проверка пропускается и в лог пишется предупреждение.

## Сборка и запуск

Локально:

```bash
# 0. Убедись, что используешь Python 3.11 (проект не тестировался на 3.14)
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Подготовка секретов
# Скопируй .env.example в .env и укажи реальные YANDEX_API_KEY и YANDEX_FOLDER_ID
cp .env.example .env

# 3. Запуск тестов
pytest tests/ -v --html=report.html --self-contained-html
```

После запуска появятся:

- `tests/ragas_results.json` — детальные результаты Ragas со средними метриками.
- `report.html` — HTML-отчёт pytest.

В CI (`PYTHONPATH` устанавливается равным `github.workspace`, чтобы `from app.rag_pipeline import pipeline` работал без установки пакета).

## Стиль кода и конвенции

- Докстринги и комментарии в проекте написаны на **русском языке** — новый код и документацию следует писать на русском.
- Используются type hints из модуля `typing` (`List`, `Dict`).
- Никаких настроек линтеров/форматеров (`black`, `ruff`, `flake8`, `mypy`) в репозитории нет.
- Минимальные изменения: проект учебный/демонстрационный, избегай избыточной инженерии.

## Инструкции по тестированию

- Для запуска тестов обязательно задать `YANDEX_API_KEY` и `YANDEX_FOLDER_ID`, иначе вызовы к YandexGPT упадут.
- Тесты делают реальные запросы к LLM и embeddings, поэтому выполнение может занимать время и тратить квоту API.
- `test_ragas_evaluation` и `test_no_hallucinations` зависят от фикстуры `ragas_client`, которая создаёт обёртки Ragas над `ChatOpenAI` и `OpenAIEmbeddings` с Yandex-совместимыми эндпоинтами.
- CI не просто полагается на `pytest` — после тестов отдельный шаг "Quality Gate Check" повторно проверяет `tests/ragas_results.json` через `jq` и `bc`.

## CI/CD и деплой

- Workflow: `.github/workflows/ci.yml`.
- Триггеры: `push` в `main`/`master`, `pull_request` в `main`/`master`, ручной запуск (`workflow_dispatch`).
- Шаги:
  1. Checkout.
  2. Установка Python 3.11.
  3. Установка зависимостей из `requirements.txt`.
  4. Запуск `pytest` с HTML-отчётом.
  5. Загрузка артефактов `ragas_results.json` и `report.html` (всегда, даже если тесты упали).
  6. Проверка quality gates по `tests/ragas_results.json`.

Деплоя нет — пайплайн только тестирует качество RAG и сохраняет артефакты.

## Безопасность

- **API-ключи и `FOLDER_ID` должны передаваться только через переменные окружения.** Локально — в файле `.env`, в GitHub Actions — в `secrets.YANDEX_API_KEY` и `secrets.YANDEX_FOLDER_ID`.
- `.env` добавлен в `.gitignore` и не должен попадать в коммиты.
- В `ragas_results.json` и `report.html` могут оказаться промпты, контексты и ответы модели — будь осторожен при публикации артефактов в публичных репозиториях.
- В `app/rag_pipeline.py` экземпляр `IMDBRAGPipeline` создаётся при импорте модуля, поэтому импорт сразу инициализирует ChromaDB-коллекцию. `OpenAI`-клиент создаётся лениво при первом вызове `generate`, чтобы не требовать `YANDEX_API_KEY` уже на этапе импорта (актуально для openai >= 2.0).
