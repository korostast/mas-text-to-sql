# Single-agent investigation update — 0.1.2

## Что изменено

Новые инструменты не добавлены. Агент использует только execute_sql и submit_sql.
Workflow prompt и одношаговое поведение сохранены. Первое answer SQL по-прежнему
фиксируется до SQL feedback. Схема, evidence, gold isolation, лимиты и usage policy
не изменены; настройки сервера в твоих configs не перезаписываются этим patch.

1. Неуспешный submit_sql в single возвращает error observation и accepted=false.
   Следующий шаг выбирает модель в оставшемся общем бюджете. Автоматического
   повторения запроса, исправления SQL или бесплатного дополнительного вызова нет.
2. execute_sql/diagnostic требует короткое поле check: что именно проверяется.
3. Изменение answer SQL требует revision с reason и sources. Допустимы question,
   schema, непустое evidence и observation:N уже исполненного SQL. Ссылка на
   ошибку исполнения допустима; ссылка на будущий/неисполненный observation — нет.
4. Модель получает рекомендации исследовать literal values, NULL, duplicates,
   joins, ties, numerator/denominator и обоснованность новых фильтров. Это выбор
   агента, не обязательная цепочка. Простой прямой ответ остаётся допустимым.
5. Новая investigation telemetry отделяет диагностику, версии кандидатов,
   основания изменений и recovery. Offline EX связывается с ней отдельно.

Runtime не имеет gold и не проверяет семантическую истинность revision.reason.
Валидная ссылка — не доказательство верности ответа. Полезность диагностики
оценивается по post-hoc repairs/regressions и ручному аудиту check/observation/revision,
а не числу вызовов. Даже EX repair после диагностики не доказывает причинность.

## Форматы

Первый простой ответ или неизменённый кандидат:

```json
{"action":"submit_sql","sql":"SELECT name FROM items"}
```

Диагностика после первого кандидата:

```json
{"action":"execute_sql","purpose":"diagnostic","check":"Check exact category labels","sql":"SELECT DISTINCT category FROM items"}
```

Изменённый ответ с основанием в наблюдении:

```json
{"action":"submit_sql","sql":"SELECT name FROM items WHERE category='Locally funded'","revision":{"reason":"Use the category label returned by the diagnostic","sources":["question","observation:2"]}}
```

Это примеры структуры, не SQL для конкретной BIRD-задачи. Даже форматирование
текста SQL требует revision (кроме внешних пробелов): AST-анализ семантических
изменений не вводился, пробелы внутри литералов не игнорируются.

## Артефакты и совместимость

Протокол: sql-investigation-v1. В run.json и episode есть protocol_version;
source hash также меняется. В single final_sql существует только у принятой
отправки. Неуспешный последний submit сохранён в last_submission. Workflow при
ошибке сохраняет прежний SQL+error артефакт; submission_accepted=false.

В investigation: diagnostics, revisions, candidate_versions, submission_attempts,
failed_submissions, recovered_submission_error. Последнее означает техническое
восстановление, а не правильный EX. successful_diagnostic_calls — SQL status=ok.
resource_summary суммирует поведение; evaluation.json/investigation_quality
добавляет repairs/regressions среди эпизодов с диагностикой. Старые артефакты
без этой телеметрии можно оценивать: coverage=0, новые неизвестные счётчики=null.

Код разделён на actions, budgets, events, generation, investigation, behavior
и EpisodeRunner. Публичные solve(...) и импорт Budget из runtime сохранены.
Начальные события и запросы записываются snapshots, без последующей мутации истории.
Изменённые функции имеют type hints и короткие English docstrings.

## Применение и проверка

Архив содержит только новые/изменённые файлы к mas-text-to-sql_v0.1.zip.
Распаковать в корень репозитория после сохранения собственных изменений.
configs, runs, данные, секреты и upstream evaluation scripts не перезаписываются.
Пакетных зависимостей не добавлено. Если используется editable installation,
при необходимости обновить её: python -m pip install -e .

```bash
python -m unittest discover -s tests -v
python -m mas_sql run --architecture single \
  --tasks local_data/split/pilot.tasks.json --db-root local_data/databases \
  --config configs/gpt-oss-usage.json --out runs/pilot_single_v012_r0 --repeat 0
python -m mas_sql evaluate \
  --tasks local_data/split/pilot.tasks.json --gold local_data/split/pilot.gold.json \
  --db-root local_data/databases --run runs/pilot_single_v012_r0 \
  --out runs/pilot_single_v012_r0/evaluation.json
```

Пути — примеры; используй свой существующий config и пути данных. Настройка
mode=usage для GPT-OSS сохранена: patch не меняет её и не даёт новой гарантии
жёсткого preflight token cap. Margin и бюджеты менять только на pilot, затем freeze.

Новый протокол меняет single prompt/поведение: запускать pilot заново в новых
каталогах. Старые single runs сохранить как baseline, не дополнять и не смешивать
с новыми. Workflow prompt проверен на байтовую идентичность прежней версии.
Локальные synthetic/mock тесты не являются результатами реальной модели; remote
vLLM и реальные BIRD эксперименты в этой реализации не запускались.
