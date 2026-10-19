# Обработка ответов и контракт действий — 0.1.4

## Исправление по результатам пилота

В 0.1.3 некорректный Chat Completions envelope приводил к общему `ClientError`:
при этом мог теряться валидный `usage`, присутствовавший рядом с неверным `content`.
Теперь эти границы разделены:

1. `llm/client.py`: один HTTP-запрос, JSON и HTTP status; без автоматического retry.
2. `llm/responses.py`: форма envelope, единственный choice, message и строковый
   content. Здесь не разбирается action JSON и не исполняется SQL.
3. `agent/generation.py`: списание доступных фактических input/output counts,
   включая ошибочные envelope, и применение прежних лимитов.
4. `agent/actions.py`: неизменный набор действий и отдельная проверка JSON action.

Исключения в `llm/errors.py` отделяют `ResponseFormatError` от прочих `ClientError`.
Существующий импорт `ClientError` из `llm.client` по-прежнему работает.

## Поведение при ошибке

| Ситуация | Учёт | Результат |
| --- | --- | --- |
| Envelope некорректен, input/output usage валиден | Списать оба счётчика ровно один раз | `response_format_error`, эпизод остановлен |
| Envelope некорректен, usage отсутствует или неполон | Сохранить известную часть; полные totals остаются `null` | `response_format_error_usage_unknown`, эпизод остановлен |
| Некорректный envelope одновременно выявил превышение фактического лимита или strict mismatch | Usage списан; проверки не отключаются | Сохраняется более приоритетная причина лимита/strict mismatch |
| HTTP/network/неразбираемый JSON | Не изобретать usage | Прежний `transport_error_usage_unknown`; точный `error_kind` в событии |
| Envelope валиден, content — строка, но это не action JSON | Usage списан штатно | Прежний protocol error и bounded correction для single |

Нулевой известный subtotal не означает бесплатный запрос. `usage_complete=false`
и `usage.*=null` сохраняют неизвестность. Пустая строка content — допустимая строка
на границе envelope, но не валидное действие. `content=null`, массив content-parts,
reasoning-only и tool-only ответы не преобразуются в действия. Reasoning никогда
не подставляется вместо content. Нативные tool calls по-прежнему не включены.

После структурной ошибки нет автоматического исправления или повторной генерации:
даже с известным usage этот эпизод завершается. Ранее проверенный кандидат не
превращается в финальный ответ. Исправление не обещает устранить серверный сбой;
оно сохраняет доступный расход и сведения для его расследования.

## Диагностика

Событие `llm_error` получает:
- `error_kind` (`response_format`, `http`, `response_json`, `transport`, `redirect`);
- при `response_format`: фиксированный `error_code`, `diagnostics`,
  `normalized_usage`, `usage_unknown` и признак tokenizer mismatch;
- для остальных ошибок: HTTP status, когда он известен.

Диагностика envelope содержит HTTP status, типы и наличие ожидаемых полей,
число choices, длину content, допустимый finish reason и ограниченный response ID.
Не записываются raw body, произвольные поля ответа, content/reasoning ошибочного
ответа, Authorization или произвольные поля объекта usage. Счётчики логируются
только в нормализованном виде. Обычные успешные ответы сохраняют прежнюю трассу
content/reasoning/usage; это не новый глобальный режим редактирования логов.

Probe также сохраняет доступный usage и структурную диагностику. Некорректный
ответ не становится успешной проверкой, даже если все counts известны; probe
останавливается после сбоя, чтобы не делать скрытых дополнительных запросов.

## Промпт

`protocol_version=sql-investigation-v1` сохранён: набор действий, первая blind
попытка и смысл метрик не менялись. `prompt_version=action-contract-v2` добавлен
в run metadata, initial event, episode, resource summary и offline report.

Для single уточнены:
- первый candidate до любой диагностики; прямой submit остаётся допустимым;
- только runtime возвращает observations; модель не имитирует результат инструмента;
- revision прикладывается к изменённому candidate/submit, а не отдельным сообщением;
- краткие полные синтаксические примеры на вымышленной `demo_sales`;
- компактные проверки сравниваемых групп и проверка обоих компонентов доли.

Нет обязательной цепочки или принудительных диагностик, новых инструментов,
данных gold в промпте, подсказок про конкретные pilot IDs и изменений budget.
Workflow prompt не изменён. Дополнительный текст single входит в его фактический
prompt usage: его стоимость не исключается из сравнения.

## Повторный пилот

Не пересоздавай split и не меняй БД. Оставь прежний рабочий config (включая
`mode=usage` и выбранный margin), endpoint и бюджеты. Используй новые каталоги,
например `runs/pilot_v014_workflow_r0` и `runs/pilot_v014_single_r0`.
Повтори probe, оба pilot run и offline evaluate по командам README, заменив
имена выходных файлов. Не смешивай новые episodes со старыми runs.

Проверь `llm_error.error_code`, `diagnostics.content_type`, `finish_reason`,
HTTP status и `normalized_usage`, если ошибка повторится. Без raw envelope
старый сбой задачи 28 нельзя точно диагностировать или восстановить его usage.

Локальные mock-тесты проверяют контракт и учёт; они не доказывают улучшения EX
или соблюдения JSON реальной моделью. Полноценный static type checker не запускался.
