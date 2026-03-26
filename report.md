# Отчет: Fact-Checking верификатор для RAG на русском языке

## 1. Датасет DRAGON

### Мотивация

Для обучения и оценки верификатора groundedness нужен датасет пар (ответ, evidence) с бинарными метками: **grounded** (ответ подтверждается evidence) и **ungrounded** (ответ не подтверждается).

### Сбор данных

Датасет собран на основе русскоязычного новостного бенчмарка DRAGON (все версии 1.0.0 — 1.15.0). Для каждого вопроса генерируются два примера:
- **grounded** — ответ модели на вопрос с релевантными evidence-текстами
- **ungrounded** — тот же ответ, но с evidence от другого, случайного вопроса

### Итоговый датасет

| Split | Grounded | Ungrounded | Всего  |
|-------|----------|------------|--------|
| train | 6 300    | 6 300      | 12 600 |
| val   | 1 350    | 1 350      | 2 700  |
| test  | 1 350    | 1 350      | 2 700  |
| **Итого** | **9 000** | **9 000** | **18 000** |

Разбиение по `question_id` (70/15/15) — один и тот же вопрос не попадает в разные сплиты.

Опубликован на HuggingFace: [`Makson4ic/dragon-derec-dataset`](https://huggingface.co/datasets/Makson4ic/dragon-derec-dataset)

## 2. Модели

### 2.1 ruDeberta (baseline)

- **Архитектура:** `deepvk/deberta-v1-base` (~124M параметров)
- **Формат входа:** `claim: question: {q}\nanswer: {a} [SEP] evidence: {e1}\n{e2}`
- **max_length:** 512 токенов
- **Проблема:** 52% примеров обрезаются при 512 токенах, теряя в среднем 32% информации из evidence

**Гиперпараметры:**
- batch_size: 8, learning_rate: 5e-6, epochs: 10, seed: 44

**Результаты на test (best epoch = 9):**

| Метрика   | Значение |
|-----------|----------|
| Accuracy  | 0.9700   |
| Precision | 0.9700   |
| Recall    | 0.9700   |
| F1        | 0.9700   |
| ROC-AUC   | 0.9928   |

Optimal threshold (Youden's J): **0.579**

Модель: [`Makson4ic/derec-dragon-ruDeberta`](https://huggingface.co/Makson4ic/derec-dragon-ruDeberta)

### 2.2 RuModernBERT-small

- **Архитектура:** `deepvk/RuModernBERT-small` (~35M параметров, max_pos=8192)
- **Формат входа:** `{question} [SEP] {answer} [SEP] {e1} [SEP] {e2}` — структурированный, с native `[SEP]` токеном (id=50282)
- **max_length:** 2048 токенов — покрывает ~96% примеров без обрезки (vs 48% при 512)

Ключевые отличия от ruDeberta:
1. **Длинный контекст** — 2048 вместо 512 токенов, минимальная потеря информации
2. **Структурированный вход** — question, answer и каждый evidence-чанк разделены native `[SEP]`, модель явно видит границы между компонентами
3. **Нет кастомных токенов** — `[SEP]` уже в словаре, не нужен `resize_token_embeddings`
4. **Компактная модель** — ~35M vs ~124M параметров

**Гиперпараметры:**
- batch_size: 8, learning_rate: 5e-6, epochs: 5, seed: 44

**Результаты на test:**

_Обучение в процессе — результаты будут добавлены после завершения._

Модель: [`Makson4ic/derec-dragon-ruModernBert-small`](https://huggingface.co/Makson4ic/derec-dragon-ruModernBert-small)

## 3. Сравнение

| Модель             | Параметры | max_length | Truncated | Accuracy | ROC-AUC |
|--------------------|-----------|------------|-----------|----------|---------|
| ruDeberta          | ~124M     | 512        | 52%       | 0.9700   | 0.9928  |
| RuModernBERT-small | ~35M      | 2048       | ~4%       | —        | —       |
