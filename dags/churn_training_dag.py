"""
dags/churn_training_dag.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Airflow DAG orchestrating the customer churn model training pipeline.

Workflow:
1. validate_raw_data:
   Loads raw dataset and verifies schema and data quality constraints using
   the project's existing validation logic.
2. train_and_benchmark:
   Invokes src.training.train:train_and_benchmark() to clean, split, train
   candidate models, log to isolated MLflow store, evaluate against test set,
   and register the best-performing model.

Guarantees:
- Schedule: None (manual trigger only).
- Catchup: False.
- Max active runs: 1.
- No DataFrames passed through XCom.
- MLflow tracking isolated via environment configuration.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from src.data.loader import load_raw
from src.data.validation import validate_raw
from src.training.train import train_and_benchmark

logger = logging.getLogger(__name__)

# Fallback shim for test environments where apache-airflow is not locally installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except (ImportError, AttributeError):

    class _BaseOperatorShim:
        def __init__(
            self,
            task_id: str,
            python_callable: Any = None,
            op_kwargs: dict[str, Any] | None = None,
            retries: int = 0,
            retry_delay: timedelta | None = None,
            dag: Any = None,
            **kwargs: Any,
        ) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            self.retries = retries
            self.retry_delay = retry_delay
            self.upstream_list: list[_BaseOperatorShim] = []
            self.downstream_list: list[_BaseOperatorShim] = []
            self.dag = dag
            if dag is not None:
                dag.tasks.append(self)
                dag.task_dict[task_id] = self

        @property
        def upstream_task_ids(self) -> set[str]:
            return {t.task_id for t in self.upstream_list}

        @property
        def downstream_task_ids(self) -> set[str]:
            return {t.task_id for t in self.downstream_list}

        def set_downstream(self, other: _BaseOperatorShim) -> _BaseOperatorShim:
            if other not in self.downstream_list:
                self.downstream_list.append(other)
            if self not in other.upstream_list:
                other.upstream_list.append(self)
            return other

        def set_upstream(self, other: _BaseOperatorShim) -> _BaseOperatorShim:
            other.set_downstream(self)
            return other

        def __rshift__(self, other: _BaseOperatorShim) -> _BaseOperatorShim:
            return self.set_downstream(other)

        def __lshift__(self, other: _BaseOperatorShim) -> _BaseOperatorShim:
            return self.set_upstream(other)

        def execute(self, context: Any = None) -> Any:
            if self.python_callable:
                return self.python_callable(**self.op_kwargs)
            return None

    class _PythonOperatorShim(_BaseOperatorShim):
        pass

    class _DAGShim:
        def __init__(
            self,
            dag_id: str,
            default_args: dict[str, Any] | None = None,
            schedule: Any = None,
            schedule_interval: Any = None,
            start_date: datetime | None = None,
            catchup: bool = False,
            max_active_runs: int = 1,
            tags: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.schedule = schedule
            self.schedule_interval = schedule_interval
            self.start_date = start_date
            self.catchup = catchup
            self.max_active_runs = max_active_runs
            self.tags = tags or []
            self.tasks: list[_BaseOperatorShim] = []
            self.task_dict: dict[str, _BaseOperatorShim] = {}

        @property
        def task_ids(self) -> list[str]:
            return list(self.task_dict.keys())

        def __enter__(self) -> _DAGShim:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def has_cycle(self) -> bool:
            visited: set[_BaseOperatorShim] = set()
            rec_stack: set[_BaseOperatorShim] = set()

            def dfs(node: _BaseOperatorShim) -> bool:
                visited.add(node)
                rec_stack.add(node)
                for neighbor in node.downstream_list:
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                rec_stack.remove(node)
                return False

            for task in self.tasks:
                if task not in visited:
                    if dfs(task):
                        return True
            return False

    DAG = _DAGShim  # type: ignore[misc,assignment]
    PythonOperator = _PythonOperatorShim  # type: ignore[misc,assignment]


DAG_ID = "customer_churn_training_pipeline"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def validate_raw_data_task() -> dict[str, Any]:
    """Validate the raw dataset using the project's data validation suite.

    Returns
    -------
    dict[str, Any]
        Validation summary containing row count and check status.

    Raises
    ------
    ValueError
        If any unexpected validation check fails.
    """
    logger.info("Executing Airflow task 'validate_raw_data'...")
    raw_df = load_raw()
    logger.info(
        "Raw dataset loaded successfully: %d rows, %d columns.",
        len(raw_df),
        len(raw_df.columns),
    )

    report = validate_raw(raw_df)
    logger.info(
        "Raw validation executed. Total checks: %d, passed: %s.",
        len(report.results),
        report.passed,
    )

    # In raw Telco dataset, TotalCharges contains known pre-cleaning whitespace strings
    # which is handled by cleaning. Any other failure is an unexpected data quality failure.
    expected_raw_failures = {"numerical_types"}
    actual_raw_failures = {f.name for f in report.failures}
    unexpected_raw_failures = actual_raw_failures - expected_raw_failures

    if unexpected_raw_failures:
        logger.error(
            "Raw dataset failed validation with unexpected failures: %s",
            unexpected_raw_failures,
        )
        report.raise_on_failure()

    logger.info(
        "Raw data validation check succeeded (expected warnings: %s).",
        list(actual_raw_failures),
    )
    return {
        "status": "success",
        "rows": int(len(raw_df)),
        "columns": int(len(raw_df.columns)),
        "expected_precleaning_issues": sorted(actual_raw_failures),
    }


def train_and_benchmark_task() -> dict[str, Any]:
    """Execute model training, benchmarking, and MLflow registration.

    Returns
    -------
    dict[str, Any]
        Summary of the winning candidate model and test evaluation results.
    """
    logger.info("Executing Airflow task 'train_and_benchmark'...")
    _, val_df, best_model_name, test_result = train_and_benchmark()

    summary = {
        "status": "success",
        "best_model_name": best_model_name,
        "best_validation_roc_auc": float(val_df.loc[best_model_name, "roc_auc"]),
        "test_roc_auc": float(test_result.roc_auc),
        "test_accuracy": float(test_result.accuracy),
        "test_f1": float(test_result.f1),
    }
    logger.info(
        "Training pipeline completed successfully. Winner: %s (Test ROC-AUC: %.4f).",
        best_model_name,
        test_result.roc_auc,
    )
    return summary


with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Customer churn end-to-end model training and MLflow benchmarking pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "churn", "training", "benchmarking"],
) as dag:
    validate_raw_data = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data_task,
        dag=dag,
    )

    train_and_benchmark = PythonOperator(
        task_id="train_and_benchmark",
        python_callable=train_and_benchmark_task,
        dag=dag,
    )

    validate_raw_data >> train_and_benchmark
