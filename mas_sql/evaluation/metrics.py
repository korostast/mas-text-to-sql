"""BIRD-compatible result metrics with explicitly preserved matching quirks."""
import statistics
from collections.abc import Hashable, Sequence
from typing import TypeAlias

Scalar: TypeAlias = str | int | float | bool | None
Value: TypeAlias = Scalar | dict[str, "Value"] | list["Value"] | tuple["Value", ...]
ResultRows: TypeAlias = Sequence[Sequence[Value]]
FrozenRow: TypeAlias = tuple[Hashable, ...]


def freeze(value: Value) -> Hashable:
    """Convert nested result values into hashable comparison keys."""
    if isinstance(value, dict):
        return tuple(sorted((key, freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def _unique_rows(rows: ResultRows) -> list[FrozenRow]:
    """Remove duplicate rows while preserving their first-seen order."""
    return list(dict.fromkeys(tuple(freeze(value) for value in row) for row in rows))


def ex_bird(predicted: ResultRows, gold: ResultRows) -> int:
    """Compare row sets, ignoring row order and multiplicity but not columns."""
    return int(set(_unique_rows(predicted)) == set(_unique_rows(gold)))


def soft_f1_bird(predicted: ResultRows, gold: ResultRows) -> float:
    """Reproduce upstream's row-order-sensitive, column-position-free SoftF1."""
    if not predicted and not gold:
        return 1.0
    predicted_rows: list[FrozenRow] = _unique_rows(predicted)
    gold_rows: list[FrozenRow] = _unique_rows(gold)
    true_positive: float = 0.0
    false_positive: float = 0.0
    false_negative: float = 0.0
    for index, truth in enumerate(gold_rows):
        if index >= len(predicted_rows):
            false_negative += 1
            continue
        if not truth:
            raise ValueError("Zero-column result unsupported")
        prediction: FrozenRow = predicted_rows[index]
        true_positive += sum(value in truth for value in prediction) / len(truth)
        false_positive += sum(value not in truth for value in prediction) / len(truth)
        false_negative += sum(value not in prediction for value in truth) / len(truth)
    false_positive += max(0, len(predicted_rows) - len(gold_rows))
    precision: float = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall: float = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rv_reward(ratio: float) -> float:
    """Map a gold/predicted timing ratio to the classic R-VES reward tiers."""
    return 1.25 if ratio >= 2 else 1.0 if ratio >= 1 else 0.75 if ratio >= 0.5 else 0.5 if ratio >= 0.25 else 0.25


def filtered_mean(values: Sequence[float]) -> float:
    """Average positive timing ratios with safe zero-variance outlier filtering."""
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected positive nonempty timing ratios")
    mean: float = statistics.mean(values)
    deviation: float = statistics.pstdev(values)
    retained: Sequence[float] = values if deviation == 0 else [v for v in values if abs(v - mean) <= 3 * deviation]
    return statistics.mean(retained or values)
