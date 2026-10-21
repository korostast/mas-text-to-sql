# MAS Text-to-SQL

Проект на базе BIRD mini-dev. Интегрированы подготовка данных, read-only SQLite, EX / SoftF1 / R_VES_safe_v1,
workflow, single-agent и последовательная MAS Author–Critic. Текущий runtime — v0.1.8.
Контракт ролей — в SEQUENTIAL_PROTOCOL.md; ограничения экспериментальной воспроизводимости
будут зафиксированы отдельным документом, не историей Git.

Python 3.11+, Linux/WSL. Новый harness использует только стандартную библиотеку.
`requirements.txt` относится к legacy BIRD-скриптам, не к mas_sql.

```bash
python -m unittest discover -s tests -v
python -m mas_sql --help
```

Данные и результаты держать в `local_data/` и `runs/`; `.env` не коммитить.
Gold SQL хранится отдельно и доступен только offline evaluator.
