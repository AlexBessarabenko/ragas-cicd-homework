# CI/CD Pipeline для LLM с Ragas и проверкой на галлюцинации

## 📋 Описание

Проект демонстрирует создание CI/CD-пайплайна для тестирования LLM-приложения (RAG) с использованием фреймворка **Ragas**. Пайплайн автоматически проверяет качество ответов, включая детекцию галлюцинаций.

### Целевое приложение
RAG-система на базе:
- **ChromaDB** — векторная база данных для хранения отзывов о фильмах
- **YandexGPT** — генерация ответов на основе извлеченного контекста

## 🎯 Метрики Ragas

| Метрика | Описание | Порог |
|---------|----------|-------|
| **Faithfulness** | Проверяет, что ответ основан на контексте (детекция галлюцинаций) | ≥ 0.7 |
| **Answer Similarity** (`semantic_similarity`) | Оценивает семантическую близость ответа и эталонного ответа | ≥ 0.6 |
| **Context Recall** | Проверяет, что контекст содержит информацию для ответа | ≥ 0.6 |

## 📁 Структура проекта
├── app/
│ ├── rag_pipeline.py # RAG-приложение
│ └── ragas_compat.py # совместимость ragas с Python 3.14 / langchain-community
├── tests/
│ ├── goldens.json # 10 эталонных примеров
│ └── test_ragas.py # Pytest + Ragas тесты
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions workflow
├── requirements.txt
└── README.md

## 🚀 Локальный запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения
Создайте файл `.env` в корне проекта:
```env
YANDEX_API_KEY=your_api_key
YANDEX_FOLDER_ID=your_folder_id
```

### 3. Запуск тестов
```bash
pytest tests/ -v --html=report.html
```

### 4. Просмотр результатов
После запуска в `tests/ragas_results.json` сохранятся детальные результаты.

## 🔧 CI/CD Pipeline

### GitHub Actions Workflow

Пайплайн автоматически запускается при:
- Push в `main`/`master`
- Pull Request
- Ручном запуске (workflow_dispatch)

### Шаги пайплайна:
1. **Checkout** — загрузка кода
2. **Setup Python** — установка Python 3.14
3. **Install dependencies** — установка зависимостей
4. **Run Ragas tests** — запуск pytest с генерацией HTML-отчета
5. **Upload artifacts** — сохранение `ragas_results.json` и `report.html`
6. **Quality Gate Check** — проверка порогов метрик

### Quality Gates
Пайплайн **падает**, если:
- Faithfulness < 0.7 (обнаружены галлюцинации)
- Answer Similarity < 0.6
- Context Recall < 0.6

### Секреты GitHub
Для работы CI/CD необходимо добавить в **Settings → Secrets and variables → Actions**:
- `YANDEX_API_KEY`
- `YANDEX_FOLDER_ID`

## 📊 Интерпретация результатов

### Faithfulness (проверка на галлюцинации)
- **≥ 0.9**: Отлично, ответ полностью основан на контексте
- **0.7–0.9**: Хорошо, минимальные отклонения
- **< 0.7**: Плохо, модель выдумывает факты

### Answer Similarity
- **≥ 0.8**: Ответ близок к эталонному
- **0.6–0.8**: Ответ частично близок
- **< 0.6**: Ответ не соответствует эталонному

### Context Recall
- **≥ 0.8**: Retrieval нашел всю нужную информацию
- **0.6–0.8**: Retrievel нашел часть информации
- **< 0.6**: Retrieval неэффективен, нужна оптимизация```
