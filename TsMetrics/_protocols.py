"""Structural contract used by forecast evaluation."""

from __future__ import annotations

from typing import Any, Protocol


class EvaluationModelProtocol(Protocol):
    """Structural estimator contract accepted by unified forecast evaluation."""

    data: Any
    _evaluation_target_name: str

    def _clone_for_evaluation(
        self,
        data: Any,
        exog: Any | None = None,
        *,
        dates: Any | None = None,
    ) -> Any:
        """Clone model configuration with isolated evaluation data."""

    def _evaluation_actual(self, observed: Any, train_data: Any) -> Any:
        """Return the observable target used for evaluation."""

    def _evaluation_predict_kwargs(
        self,
        start: int,
        stop: int,
    ) -> dict[str, Any]:
        """Return model-specific prediction context."""

    def _validate_evaluation(self, context: str) -> None:
        """Validate model-specific evaluation requirements."""
