# Структура mas_sql — 0.1.7

Дополнение v0.1.7:

- `agent/sequential/grounding.py` — прямые цитаты, тип основания, свежесть/происхождение наблюдений; без семантического оракула.
- `agent/sequential/closure.py` — чистая арифметика оценочного резерва до явного submit.
- `GenerationController` — opt-in holdback/output floor; defaults для workflow/single прежние.
- Полный новый контракт — `SEQUENTIAL_PROTOCOL.md`, ограничения — `GROUNDS_CLOSURE.md`.

## Разделение по ответственности

```text
mas_sql/
├── __init__.py              # версия пакета
├── __main__.py              # python -m mas_sql
├── cli.py                   # команды и orchestration верхнего уровня
├── agent/
│   ├── actions.py           # разбор и проверка действий модели
│   ├── budgets.py           # бюджеты эпизода
│   ├── events.py            # журнал событий
│   ├── generation.py        # вызов генерации и обработка ответа
│   ├── investigation.py     # диагностические проверки и основания ревизий
│   ├── prompts.py           # промпты workflow/single
│   └── runtime.py           # выполнение эпизода
├── llm/
│   ├── client.py            # HTTP-клиент Chat Completions и /tokenize
│   ├── errors.py            # типизированные сбои с независимым usage
│   ├── responses.py         # проверка envelope и безопасная диагностика
│   ├── accounting.py        # preflight и фактический usage
│   ├── types.py             # типы сообщений и ответов
│   └── probe.py             # проверка возможностей сервера
├── database/
│   ├── executor.py          # запуск read-only SQL, preview и DDL
│   ├── worker.py            # SQLite в изолированном дочернем процессе
│   └── paths.py             # безопасное разрешение путей БД
├── data/
│   ├── dataset.py           # типы записей, валидация, выборка и prepare
│   ├── io.py                # JSON, атомарная запись и SHA-256
│   └── manifest.py          # проверка замороженных task/gold/DB
└── evaluation/
    ├── evaluator.py         # offline scoring и объединение по task ID
    ├── metrics.py           # EX, SoftF1 и R-VES helpers
    └── behavior.py          # агрегаты поведения агента
```

В каждой папке также есть минимальный `__init__.py`. Реализация не
реэкспортируется через них: импорты показывают фактический модуль, а импорт
пакета не запускает CLI или worker. Классы остаются там, где уже есть состояние
или конфигурация; чистые метрики и операции с файлами остаются функциями.

## Зависимости

- `cli` связывает подготовку данных, запуск, probe и оценку.
- `agent` использует `llm` и `database`, но не offline evaluator.
- `evaluation` использует `data` и `database`; gold не нужен runtime агента.
- `data` использует только `database.paths` для поиска файлов SQLite,
  не зависит от оценки или генерации. Проверка manifest вынесена сюда,
  поэтому для проверки входа не требуется импорт evaluator.
- `llm` и `database` не импортируют агентный runtime.

`common.py` удалён: файловый I/O, пути БД и валидация task больше не смешаны
в одном модуле. Старые плоские модули не оставлены как compatibility shims.

## Совместимость

Команды `python -m mas_sql` и `mas-sql`, их аргументы, конфиги и протокол
`sql-investigation-v1` сохранены. Новый путь дочернего SQL-процесса —
`mas_sql.database.worker`. Хеш исходников рекурсивно включает все `.py`
внутри `mas_sql` вместе с относительными путями.

Прямые Python-импорты требуют обновления. Например:

```python
from mas_sql.agent.runtime import Budget, solve
from mas_sql.data.dataset import prepare, validate_tasks
from mas_sql.database.executor import SQLLimits, execute
from mas_sql.evaluation.metrics import ex_bird, soft_f1_bird
from mas_sql.llm.client import ChatClient
```

Переносы: `data.py` → `data/dataset.py`, `evaluate.py` →
`evaluation/evaluator.py`, `sql.py` → `database/executor.py`,
`tokens.py` → `llm/accounting.py`. Остальные модули перенесены под тем же
именем в соответствующую папку. `verify_manifest` теперь в `data/manifest.py`.

## Аннотации и документация

Все функции, включая вложенные, имеют аннотации аргументов и возвращаемых
значений; `self`/`cls` не требуют отдельных аннотаций. Все модули, классы и
функции имеют короткие английские docstrings. Для записей датасета и manifest
добавлены `TypedDict`, для значений метрик — именованные типы; локальные
коллекции и неоднозначные переменные аннотированы явно. На динамических
JSON/HTTP-границах остаётся `Any` с последующей runtime-валидацией.

`tests/test_structure.py` проверяет структуру корня, импорты всех модулей,
наличие аннотаций/docstrings и рекурсивный хеш. Это проверка AST и поведения,
а не замена полноценному статическому анализатору типов.

## Дополнение 0.1.4

Разбор HTTP/JSON, проверка Chat Completions envelope и парсинг action JSON
разделены. `ResponseFormatError` сохраняет usage для учёта, но не разрешает
выполнение SQL из некорректного ответа. Подробнее в `RESPONSE_HANDLING.md`.


## Дополнение v0.1.5: последовательная MAS

`mas_sql/agent/sequential/` содержит `actions.py`, `state.py`, `prompts.py`,
`roles.py`, `runtime.py`. Пакет изолирует новый протокол от прежнего single-agent.
Точные границы и state machine описаны в `SEQUENTIAL_PROTOCOL.md`.
`agent/runtime.solve` диспетчеризует три архитектуры; accounting и SQLite общие.
`evaluation/behavior.py` и `evaluator.py` дополнены счётчиками взаимодействия
и офлайн-оценкой конкретных проверенных версий SQL, без изменения EX/SoftF1/R-VES.


## Дополнение v0.1.6: grounding и публичные снимки

`sequential/prompts.py` собирает один source-preserving snapshot;
`sequential/roles.py` пересобирает контекст из исходной задачи, текущего snapshot
и максимум одной приватной пары protocol-error/feedback. Вся публичная история
по-прежнему хранится в `EpisodeState`, версии и источники не удаляются.
`state.available_sources` — общий реестр для context packet и валидатора.
`runtime.py` публикует `critic_review.review_id`, CLI резервирует JSONL `id`
для задачи. `context_version` фиксируется в run/resource/episode/evaluation.
Подробности — `GROUNDING_CONTEXT.md`, контракт — `SEQUENTIAL_PROTOCOL.md`.
