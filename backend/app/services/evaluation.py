import logging
from typing import Optional
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    TrainingExperiment, EvaluationResult, ComparisonStudy,
    TaskStatus, TrainingMode
)
from ..services.training import execute_training, TextDataset, TRAINER_MAP, _oversample_minority
from ..models.db_models import Sample, SplitType, DatasetVersion

logger = logging.getLogger(__name__)


async def get_evaluation(session: AsyncSession, experiment_id: int) -> Optional[dict]:
    stmt = select(EvaluationResult).where(EvaluationResult.experiment_id == experiment_id)
    result = await session.execute(stmt)
    eval_result = result.scalar_one_or_none()
    if not eval_result:
        return None
    return {
        "id": eval_result.id,
        "experiment_id": eval_result.experiment_id,
        "accuracy": eval_result.accuracy,
        "macro_f1": eval_result.macro_f1,
        "weighted_f1": eval_result.weighted_f1,
        "per_class_metrics": eval_result.per_class_metrics,
        "confusion_matrix": eval_result.confusion_matrix,
        "created_at": eval_result.created_at.isoformat() if eval_result.created_at else None,
    }


async def run_learning_curve(
    session: AsyncSession,
    dataset_id: int,
    version_id: int,
    backbone,
    training_mode: TrainingMode,
    data_fractions: list[float],
    hyperparams: dict,
) -> list[dict]:
    stmt = select(Sample).where(
        Sample.version_id == version_id,
        Sample.is_filtered == False,
        Sample.split == SplitType.train,
    )
    result = await session.execute(stmt)
    train_samples = result.scalars().all()

    val_stmt = select(Sample).where(
        Sample.version_id == version_id,
        Sample.is_filtered == False,
        Sample.split == SplitType.val,
    )
    val_result = await session.execute(val_stmt)
    val_samples = val_result.scalars().all()

    test_stmt = select(Sample).where(
        Sample.version_id == version_id,
        Sample.is_filtered == False,
        Sample.split == SplitType.test,
    )
    test_result = await session.execute(test_stmt)
    test_samples = test_result.scalars().all()

    if not train_samples:
        return []

    all_labels = sorted(set(s.label for s in train_samples + val_samples + test_samples))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    results = []
    for frac in data_fractions:
        n = max(1, int(len(train_samples) * frac))
        import random
        subset = random.sample(train_samples, min(n, len(train_samples)))

        train_texts = [s.text for s in subset]
        train_labels = [s.label for s in subset]
        train_texts, train_labels = _oversample_minority(train_texts, train_labels)

        val_texts = [s.text for s in val_samples]
        val_labels = [s.label for s in val_samples]
        test_texts = [s.text for s in test_samples]
        test_labels = [s.label for s in test_samples]

        trainer_cls = TRAINER_MAP.get(backbone)
        if not trainer_cls:
            continue

        trainer = trainer_cls(
            num_classes=len(all_labels),
            label2id=label2id,
            id2label=id2label,
            hyperparams=dict(hyperparams),
        )

        train_ds = TextDataset(train_texts, train_labels, label2id)
        val_ds = TextDataset(val_texts, val_labels, label2id)
        test_ds = TextDataset(test_texts, test_labels, label2id)

        try:
            train_result = await trainer.train(train_ds, val_ds)
            if train_result and train_result.get("model_path"):
                trainer.hyperparams["_model_path"] = train_result["model_path"]
                eval_result = await trainer.evaluate(test_ds)
                results.append({
                    "fraction": frac,
                    "train_size": n,
                    **eval_result,
                })
        except Exception as e:
            logger.warning(f"Learning curve point failed for fraction {frac}: {e}")
            results.append({
                "fraction": frac,
                "train_size": n,
                "error": str(e),
            })

    return results


async def compare_strategies(
    session: AsyncSession,
    experiment_ids: list[int],
) -> dict:
    stmt = select(EvaluationResult).where(EvaluationResult.experiment_id.in_(experiment_ids))
    result = await session.execute(stmt)
    eval_results = result.scalars().all()

    exp_stmt = select(TrainingExperiment).where(TrainingExperiment.id.in_(experiment_ids))
    exp_result = await session.execute(exp_stmt)
    experiments = {e.id: e for e in exp_result.scalars().all()}

    comparison = []
    for eval_r in eval_results:
        exp = experiments.get(eval_r.experiment_id)
        if not exp:
            continue
        comparison.append({
            "experiment_id": eval_r.experiment_id,
            "experiment_name": exp.experiment_name,
            "training_mode": exp.training_mode.value,
            "backbone": exp.backbone.value,
            "accuracy": eval_r.accuracy,
            "macro_f1": eval_r.macro_f1,
            "weighted_f1": eval_r.weighted_f1,
            "per_class_metrics": eval_r.per_class_metrics,
        })

    return {"comparisons": comparison}


async def significance_test(
    session: AsyncSession,
    experiment_id_a: int,
    experiment_id_b: int,
    test_type: str = "paired_t",
    num_bootstrap: int = 10000,
) -> dict:
    stmt = select(EvaluationResult).where(
        EvaluationResult.experiment_id.in_([experiment_id_a, experiment_id_b])
    )
    result = await session.execute(stmt)
    eval_results = {r.experiment_id: r for r in result.scalars().all()}

    if experiment_id_a not in eval_results or experiment_id_b not in eval_results:
        raise ValueError("One or both experiment evaluation results not found")

    ea = eval_results[experiment_id_a]
    eb = eval_results[experiment_id_b]

    metrics_a = _extract_per_class_f1(ea.per_class_metrics)
    metrics_b = _extract_per_class_f1(eb.per_class_metrics)

    if len(metrics_a) != len(metrics_b) or len(metrics_a) < 2:
        return {
            "test_type": test_type,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note": "Insufficient data points for significance testing",
        }

    a_arr = np.array(metrics_a)
    b_arr = np.array(metrics_b)
    diff = a_arr - b_arr

    if test_type == "paired_t":
        t_stat, p_value = scipy_stats.ttest_rel(a_arr, b_arr)
        return {
            "test_type": "paired_t",
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < 0.05,
            "mean_diff": float(np.mean(diff)),
        }
    elif test_type == "bootstrap":
        bootstrap_diffs = []
        rng = np.random.default_rng(42)
        for _ in range(num_bootstrap):
            idx = rng.choice(len(diff), size=len(diff), replace=True)
            bootstrap_diffs.append(np.mean(diff[idx]))

        ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
        ci_upper = float(np.percentile(bootstrap_diffs, 97.5))
        significant = not (ci_lower <= 0 <= ci_upper)

        return {
            "test_type": "bootstrap",
            "statistic": float(np.mean(bootstrap_diffs)),
            "p_value": None,
            "significant": significant,
            "confidence_interval": [ci_lower, ci_upper],
            "mean_diff": float(np.mean(diff)),
        }

    return {"test_type": test_type, "statistic": 0.0, "p_value": 1.0, "significant": False}


def _extract_per_class_f1(per_class_metrics: dict) -> list[float]:
    if not per_class_metrics:
        return []
    return [v.get("f1-score", 0.0) for v in per_class_metrics.values() if isinstance(v, dict)]
