import logging
import time
import os
from collections import OrderedDict, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import MODEL_CACHE_DIR
from ..models.db_models import (
    MLTrainingTask,
    MLTrainingReport,
    MLTrainingStatus,
    MLModelType,
    DatasetVersion,
    Dataset,
    Sample,
    AnnotationQueue,
    FilterTask,
)

logger = logging.getLogger(__name__)

MODEL_FILE_DIR = MODEL_CACHE_DIR / "ml_models"
MODEL_FILE_DIR.mkdir(parents=True, exist_ok=True)


class LRUModelCache:
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self._cache: "OrderedDict[int, tuple]" = OrderedDict()

    def get(self, task_id: int):
        if task_id not in self._cache:
            return None
        self._cache.move_to_end(task_id)
        return self._cache[task_id]

    def put(self, task_id: int, pipeline, label_encoder: dict, id_to_label: dict):
        if task_id in self._cache:
            self._cache.move_to_end(task_id)
        else:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)
        self._cache[task_id] = (pipeline, label_encoder, id_to_label)

    def clear(self):
        self._cache.clear()


_model_cache = LRUModelCache(capacity=3)


def get_model_cache() -> LRUModelCache:
    return _model_cache


async def create_ml_training_task(
    session: AsyncSession,
    task_name: str,
    annotated_version_id: int,
    model_type: MLModelType,
    hyperparams: dict,
    split_ratios: dict,
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> MLTrainingTask:
    version_stmt = select(DatasetVersion).where(DatasetVersion.id == annotated_version_id)
    version_result = await session.execute(version_stmt)
    version = version_result.scalar_one_or_none()
    if not version:
        raise ValueError(f"Dataset version {annotated_version_id} not found")

    if version.version_type != "annotated":
        raise ValueError(
            f"Only annotated versions can be used for ML training. "
            f"Current version type: {version.version_type}"
        )

    _validate_split_ratios(split_ratios)

    validated_tags = _validate_tags(tags) if tags else []

    task = MLTrainingTask(
        task_name=task_name,
        dataset_id=version.dataset_id,
        annotated_version_id=annotated_version_id,
        model_type=model_type,
        hyperparams=hyperparams,
        split_ratios=split_ratios,
        status=MLTrainingStatus.pending,
        notes=notes,
        tags=validated_tags,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def list_ml_training_tasks(
    session: AsyncSession,
    dataset_id: Optional[int] = None,
    status: Optional[MLTrainingStatus] = None,
    tag: Optional[str] = None,
) -> list[MLTrainingTask]:
    stmt = select(MLTrainingTask).order_by(MLTrainingTask.created_at.desc())
    if dataset_id is not None:
        stmt = stmt.where(MLTrainingTask.dataset_id == dataset_id)
    if status is not None:
        stmt = stmt.where(MLTrainingTask.status == status)
    result = await session.execute(stmt)
    tasks = list(result.scalars().all())
    if tag is not None:
        tasks = [t for t in tasks if t.tags and tag in t.tags]
    return tasks


async def get_ml_training_task(
    session: AsyncSession,
    task_id: int,
) -> Optional[MLTrainingTask]:
    stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


def _build_pipeline(model_type: MLModelType, hyperparams: dict, num_classes: int = 2) -> Pipeline:
    ngram_min = int(hyperparams.get("ngram_min", 1))
    ngram_max = int(hyperparams.get("ngram_max", 1))
    if ngram_min > ngram_max:
        ngram_min, ngram_max = ngram_max, ngram_min

    vectorizer = TfidfVectorizer(ngram_range=(ngram_min, ngram_max))

    if model_type == MLModelType.naive_bayes:
        alpha = float(hyperparams.get("alpha", 1.0))
        clf = MultinomialNB(alpha=alpha)
    else:
        max_iter = int(hyperparams.get("max_iter", 100))
        C = float(hyperparams.get("C", 1.0))
        if num_classes > 2:
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=42,
            )
        else:
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="liblinear",
                random_state=42,
            )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def _validate_split_ratios(split_ratios: dict) -> None:
    if not split_ratios:
        raise ValueError("split_ratios is required")

    train_ratio = split_ratios.get("train_ratio")
    val_ratio = split_ratios.get("val_ratio")
    test_ratio = split_ratios.get("test_ratio")

    if train_ratio is None or val_ratio is None or test_ratio is None:
        raise ValueError("train_ratio, val_ratio and test_ratio are all required")

    for name, val in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be a number")
        if val <= 0:
            raise ValueError(f"{name} must be > 0")
        if val >= 1:
            raise ValueError(f"{name} must be < 1")

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Sum of split ratios must equal 1.0, got train={train_ratio} + val={val_ratio} + test={test_ratio} = {total}"
        )

    if float(train_ratio) < 0.2:
        raise ValueError("train_ratio must be at least 0.2 (need enough data to train)")
    if float(test_ratio) < 0.05:
        raise ValueError("test_ratio must be at least 0.05 for meaningful evaluation")


def _validate_tags(tags: list[str]) -> list[str]:
    if len(tags) > 10:
        raise ValueError("Maximum 10 tags allowed")
    for t in tags:
        if len(t) > 30:
            raise ValueError(f"Tag '{t[:10]}...' exceeds 30 character limit")
    return tags


async def execute_ml_training(session: AsyncSession, task_id: int) -> None:
    stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        return

    if task.status == MLTrainingStatus.cancelled:
        return

    if task.status != MLTrainingStatus.pending:
        return

    task.status = MLTrainingStatus.training
    task.started_at = datetime.utcnow()
    await session.commit()

    start_time = time.time()

    try:
        sample_stmt = select(Sample).where(
            Sample.version_id == task.annotated_version_id,
            Sample.is_filtered == False,
        )
        sample_result = await session.execute(sample_stmt)
        samples = list(sample_result.scalars().all())

        if len(samples) < 2:
            raise ValueError("Insufficient samples for training (need at least 2)")

        texts = [s.text for s in samples]
        labels = [s.label for s in samples]

        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 distinct classes for training")

        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        y = np.array([label_to_id[l] for l in labels])

        ratios = task.split_ratios or {}
        train_ratio = ratios.get("train_ratio", 0.7)
        val_ratio = ratios.get("val_ratio", 0.15)
        test_ratio = ratios.get("test_ratio", 0.15)

        test_size_adj = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
        val_test_size = 1.0 - train_ratio

        X_train, X_val_test, y_train, y_val_test = train_test_split(
            texts, y, test_size=val_test_size, random_state=42, stratify=y
        )
        if len(X_val_test) > 0 and val_test_size > 0 and test_size_adj > 0 and test_size_adj < 1:
            try:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_val_test, y_val_test, test_size=test_size_adj, random_state=42, stratify=y_val_test
                )
            except ValueError:
                X_val, X_test, y_val, y_test = X_val_test, [], y_val_test, np.array([], dtype=int)
        else:
            X_val, X_test, y_val, y_test = X_val_test, [], y_val_test, np.array([], dtype=int)

        pipeline = _build_pipeline(task.model_type, task.hyperparams or {}, num_classes=len(unique_labels))
        pipeline.fit(X_train, y_train)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        try:
            train_pred = pipeline.predict(X_train)
            train_proba = pipeline.predict_proba(X_train)
            train_acc = float(accuracy_score(y_train, train_pred))
            try:
                train_loss = float(log_loss(y_train, train_proba, labels=list(range(len(unique_labels)))))
            except Exception:
                train_loss = 0.0
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        except Exception:
            pass

        if len(X_val) > 0:
            try:
                val_pred = pipeline.predict(X_val)
                val_proba = pipeline.predict_proba(X_val)
                val_acc = float(accuracy_score(y_val, val_pred))
                try:
                    val_loss = float(log_loss(y_val, val_proba, labels=list(range(len(unique_labels)))))
                except Exception:
                    val_loss = 0.0
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            except Exception:
                pass

        model_filename = f"ml_task_{task.id}_{int(time.time())}.joblib"
        model_path = MODEL_FILE_DIR / model_filename
        model_data = {
            "pipeline": pipeline,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "class_names": unique_labels,
            "task_id": task.id,
        }
        joblib.dump(model_data, str(model_path))
        model_size = os.path.getsize(str(model_path))

        if len(X_test) > 0:
            test_pred = pipeline.predict(X_test)
            test_proba = pipeline.predict_proba(X_test)
            test_pred_labels = [id_to_label[int(p)] for p in test_pred]
            y_test_labels = [id_to_label[int(p)] for p in y_test]

            accuracy = float(accuracy_score(y_test_labels, test_pred_labels))
            weighted_f1 = float(f1_score(y_test_labels, test_pred_labels, average="weighted", zero_division=0))

            report = classification_report(
                y_test_labels, test_pred_labels, output_dict=True, zero_division=0
            )
            per_class = {}
            total_support = sum(
                int(metrics.get("support", 0))
                for key, metrics in report.items()
                if key not in ("accuracy", "macro avg", "weighted avg")
            )
            pred_counter = Counter(test_pred_labels)
            actual_counter = Counter(y_test_labels)

            for key, metrics in report.items():
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                support_val = int(metrics.get("support", 0))
                support_ratio = support_val / total_support if total_support > 0 else 0.0
                actual_count = actual_counter.get(key, 0)
                pred_count = pred_counter.get(key, 0)
                prediction_bias = pred_count / actual_count if actual_count > 0 else None

                per_class[key] = {
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1-score": float(metrics.get("f1-score", 0.0)),
                    "support": support_val,
                    "support_ratio": round(support_ratio, 6),
                    "prediction_bias": round(prediction_bias, 6) if prediction_bias is not None else None,
                }

            cm = confusion_matrix(y_test_labels, test_pred_labels, labels=unique_labels)
            confusion_matrix_list = cm.tolist()

            roc_auc = None
            if len(unique_labels) == 2:
                try:
                    roc_auc = float(roc_auc_score(y_test, test_proba[:, 1]))
                except Exception:
                    roc_auc = None

            report_row = MLTrainingReport(
                task_id=task.id,
                accuracy=accuracy,
                weighted_f1=weighted_f1,
                per_class_metrics=per_class,
                confusion_matrix=confusion_matrix_list,
                class_names=unique_labels,
                roc_auc=roc_auc,
            )
            session.add(report_row)

        duration = time.time() - start_time

        cancelled_stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
        cancelled_result = await session.execute(cancelled_stmt)
        refreshed_task = cancelled_result.scalar_one_or_none()
        if refreshed_task and refreshed_task.status == MLTrainingStatus.cancelled:
            await session.commit()
            return

        task.status = MLTrainingStatus.completed
        task.model_path = str(model_path)
        task.model_size_bytes = int(model_size)
        task.training_duration_seconds = round(duration, 3)
        task.train_loss_history = train_losses
        task.train_acc_history = train_accs
        task.val_loss_history = val_losses
        task.val_acc_history = val_accs
        task.completed_at = datetime.utcnow()

        await session.commit()

        cache = get_model_cache()
        cache.put(task.id, pipeline, label_to_id, id_to_label)

    except Exception as e:
        logger.exception(f"ML training task {task_id} failed")
        task.status = MLTrainingStatus.failed
        task.error_message = str(e)
        task.completed_at = datetime.utcnow()
        task.training_duration_seconds = round(time.time() - start_time, 3)
        await session.commit()


async def get_ml_training_report(
    session: AsyncSession,
    task_id: int,
) -> Optional[MLTrainingReport]:
    stmt = select(MLTrainingReport).where(MLTrainingReport.task_id == task_id)
    result = await session.execute(stmt)
    report = result.scalar_one_or_none()
    if not report:
        return None

    per_class = report.per_class_metrics or {}
    enriched = {}
    for class_name, metrics in per_class.items():
        enriched[class_name] = {
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1-score": metrics.get("f1-score", 0.0),
            "support": metrics.get("support", 0),
            "support_ratio": metrics.get("support_ratio"),
            "prediction_bias": metrics.get("prediction_bias"),
        }
    report.per_class_metrics = enriched

    return report


async def compare_ml_models(
    session: AsyncSession,
    task_ids: list[int],
) -> dict:
    if not task_ids:
        raise ValueError("No task_ids provided")

    stmt = select(MLTrainingTask).where(MLTrainingTask.id.in_(task_ids))
    result = await session.execute(stmt)
    tasks = {t.id: t for t in result.scalars().all()}

    missing = [tid for tid in task_ids if tid not in tasks]
    if missing:
        raise ValueError(f"Task(s) not found: {missing}")

    dataset_ids = set(t.dataset_id for t in tasks.values())
    if len(dataset_ids) > 1:
        raise ValueError(
            f"All tasks must belong to the same dataset. Found datasets: {sorted(dataset_ids)}"
        )

    dataset_id = next(iter(dataset_ids))
    ds_stmt = select(Dataset).where(Dataset.id == dataset_id)
    ds_result = await session.execute(ds_stmt)
    dataset = ds_result.scalar_one_or_none()
    dataset_name = dataset.name if dataset else "unknown"

    report_stmt = select(MLTrainingReport).where(MLTrainingReport.task_id.in_(task_ids))
    report_result = await session.execute(report_stmt)
    reports = {r.task_id: r for r in report_result.scalars().all()}

    invalid = [tid for tid in task_ids if tasks[tid].status != MLTrainingStatus.completed]
    if invalid:
        raise ValueError(f"Task(s) not completed: {invalid}")

    items = []
    for tid in task_ids:
        task = tasks[tid]
        report = reports.get(tid)
        per_class = report.per_class_metrics if report else {}
        items.append({
            "task_id": task.id,
            "task_name": task.task_name,
            "model_type": task.model_type.value,
            "accuracy": report.accuracy if report else 0.0,
            "weighted_f1": report.weighted_f1 if report else 0.0,
            "training_duration_seconds": task.training_duration_seconds,
            "model_size_bytes": task.model_size_bytes,
            "per_class_metrics": per_class,
        })

    delta_summary = None
    if len(items) >= 2:
        best_task_id = None
        best_metrics = []
        best_score = -1.0

        metric_keys = ["accuracy", "weighted_f1"]
        for item in items:
            score = sum(item.get(k, 0.0) for k in metric_keys)
            if score > best_score:
                best_score = score
                best_task_id = item["task_id"]

        if best_task_id is not None:
            best_item = next(i for i in items if i["task_id"] == best_task_id)
            for key in metric_keys:
                best_val = best_item.get(key, 0.0)
                is_best = all(i.get(key, 0.0) <= best_val for i in items if i["task_id"] != best_task_id)
                if is_best:
                    best_metrics.append(f"{key}_best")

            delta_summary = {
                "best_task_id": best_task_id,
                "best_metrics": best_metrics,
            }
    elif len(items) == 1:
        delta_summary = None

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "items": items,
        "delta_summary": delta_summary,
    }


async def cancel_ml_training_task(
    session: AsyncSession,
    task_id: int,
) -> MLTrainingTask:
    stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    if task.status not in (MLTrainingStatus.pending, MLTrainingStatus.training):
        raise ValueError(
            f"Cannot cancel task in '{task.status.value}' status. "
            f"Only pending or training tasks can be cancelled."
        )
    task.status = MLTrainingStatus.cancelled
    task.cancelled_at = datetime.utcnow()
    await session.commit()
    await session.refresh(task)
    return task


async def retry_ml_training_task(
    session: AsyncSession,
    task_id: int,
) -> MLTrainingTask:
    stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
    result = await session.execute(stmt)
    original_task = result.scalar_one_or_none()
    if not original_task:
        raise ValueError(f"Training task {task_id} not found")
    if original_task.status != MLTrainingStatus.failed:
        raise ValueError(
            f"Cannot retry task in '{original_task.status.value}' status. "
            f"Only failed tasks can be retried."
        )

    new_task = MLTrainingTask(
        task_name=f"{original_task.task_name} (retry)",
        dataset_id=original_task.dataset_id,
        annotated_version_id=original_task.annotated_version_id,
        model_type=original_task.model_type,
        hyperparams=dict(original_task.hyperparams) if original_task.hyperparams else {},
        split_ratios=dict(original_task.split_ratios) if original_task.split_ratios else {},
        status=MLTrainingStatus.pending,
        retry_from=original_task.id,
        notes=original_task.notes,
        tags=list(original_task.tags) if original_task.tags else [],
    )
    session.add(new_task)
    await session.commit()
    await session.refresh(new_task)
    return new_task


async def update_ml_training_task_metadata(
    session: AsyncSession,
    task_id: int,
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> MLTrainingTask:
    stmt = select(MLTrainingTask).where(MLTrainingTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise ValueError(f"Training task {task_id} not found")

    if notes is not None:
        if len(notes) > 500:
            raise ValueError("Notes must not exceed 500 characters")
        task.notes = notes

    if tags is not None:
        validated_tags = _validate_tags(tags)
        task.tags = validated_tags

    await session.commit()
    await session.refresh(task)
    return task


async def _load_model_for_task(task: MLTrainingTask):
    cache = get_model_cache()
    cached = cache.get(task.id)
    if cached is not None:
        return cached

    if not task.model_path or not Path(task.model_path).exists():
        raise ValueError("Model file not found on disk")

    model_data = joblib.load(task.model_path)
    pipeline = model_data["pipeline"]
    label_to_id = model_data["label_to_id"]
    id_to_label = model_data["id_to_label"]

    cache.put(task.id, pipeline, label_to_id, id_to_label)
    return pipeline, label_to_id, id_to_label


async def predict_single(session: AsyncSession, task_id: int, text: str) -> dict:
    task = await get_ml_training_task(session, task_id)
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    if task.status != MLTrainingStatus.completed:
        raise ValueError(
            f"Model not ready. Task status is '{task.status.value}', expected 'completed'."
        )

    pipeline, label_to_id, id_to_label = await _load_model_for_task(task)

    pred_idx = int(pipeline.predict([text])[0])
    proba = pipeline.predict_proba([text])[0]

    probabilities = {}
    for idx, prob in enumerate(proba):
        label = id_to_label.get(idx, str(idx))
        probabilities[label] = round(float(prob), 6)

    predicted_label = id_to_label.get(pred_idx, str(pred_idx))
    return {
        "task_id": task_id,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
    }


async def predict_batch(session: AsyncSession, task_id: int, texts: list[str]) -> dict:
    if len(texts) > 100:
        raise ValueError("Batch prediction supports at most 100 texts")

    task = await get_ml_training_task(session, task_id)
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    if task.status != MLTrainingStatus.completed:
        raise ValueError(
            f"Model not ready. Task status is '{task.status.value}', expected 'completed'."
        )

    pipeline, label_to_id, id_to_label = await _load_model_for_task(task)

    safe_texts = [t if t else "" for t in texts]
    preds = pipeline.predict(safe_texts)
    probas = pipeline.predict_proba(safe_texts)

    results = []
    for text, pred_idx, proba in zip(texts, preds, probas):
        probabilities = {}
        for idx, p in enumerate(proba):
            label = id_to_label.get(idx, str(idx))
            probabilities[label] = round(float(p), 6)
        predicted_label = id_to_label.get(int(pred_idx), str(int(pred_idx)))
        results.append({
            "text": text,
            "predicted_label": predicted_label,
            "probabilities": probabilities,
        })

    return {"task_id": task_id, "results": results}


async def get_data_lineage(session: AsyncSession, task_id: int) -> dict:
    task = await get_ml_training_task(session, task_id)
    if not task:
        raise ValueError(f"Training task {task_id} not found")

    chain = []

    chain.append({
        "node_type": "ml_training_task",
        "id": task.id,
        "name": task.task_name,
        "info": {
            "model_type": task.model_type.value,
            "status": task.status.value,
            "hyperparams": task.hyperparams or {},
            "split_ratios": task.split_ratios or {},
            "created_at": task.created_at.isoformat() if task.created_at else None,
        },
    })

    annotated_version_stmt = select(DatasetVersion).where(DatasetVersion.id == task.annotated_version_id)
    annotated_result = await session.execute(annotated_version_stmt)
    annotated_version = annotated_result.scalar_one_or_none()
    if annotated_version:
        chain.append({
            "node_type": "annotated_version",
            "id": annotated_version.id,
            "name": annotated_version.version_name,
            "info": {
                "version_type": annotated_version.version_type,
                "total_samples": annotated_version.total_samples,
                "class_distribution": annotated_version.class_distribution or {},
                "parent_version_id": annotated_version.parent_version_id,
                "created_at": annotated_version.created_at.isoformat() if annotated_version.created_at else None,
            },
        })

        queue_stmt = select(AnnotationQueue).where(
            AnnotationQueue.target_version_id == annotated_version.id
        )
        queue_result = await session.execute(queue_stmt)
        queue = queue_result.scalar_one_or_none()
        if queue:
            chain.append({
                "node_type": "annotation_queue",
                "id": queue.id,
                "name": queue.name,
                "info": {
                    "status": queue.status.value,
                    "capacity": queue.capacity,
                    "priority_strategy": queue.priority_strategy.value,
                    "review_mode": queue.review_mode,
                    "num_reviewers": queue.num_reviewers,
                    "created_at": queue.created_at.isoformat() if queue.created_at else None,
                },
            })

            filtered_version_stmt = select(DatasetVersion).where(DatasetVersion.id == queue.version_id)
            filtered_result = await session.execute(filtered_version_stmt)
            filtered_version = filtered_result.scalar_one_or_none()
            if filtered_version:
                chain.append({
                    "node_type": "filtered_version",
                    "id": filtered_version.id,
                    "name": filtered_version.version_name,
                    "info": {
                        "version_type": filtered_version.version_type,
                        "total_samples": filtered_version.total_samples,
                        "filter_strictness": filtered_version.filter_strictness,
                        "parent_version_id": filtered_version.parent_version_id,
                        "created_at": filtered_version.created_at.isoformat() if filtered_version.created_at else None,
                    },
                })

                filter_task_stmt = select(FilterTask).where(
                    FilterTask.target_version_id == filtered_version.id
                )
                filter_task_result = await session.execute(filter_task_stmt)
                filter_task = filter_task_result.scalar_one_or_none()
                if filter_task:
                    chain.append({
                        "node_type": "filter_task",
                        "id": filter_task.id,
                        "name": f"filter_task_{filter_task.id}",
                        "info": {
                            "strictness": filter_task.strictness.value if hasattr(filter_task.strictness, "value") else str(filter_task.strictness),
                            "status": filter_task.status.value if hasattr(filter_task.status, "value") else str(filter_task.status),
                            "total_samples": filter_task.total_samples,
                            "passed_samples": filter_task.passed_samples,
                            "filtered_samples": filter_task.filtered_samples,
                            "created_at": filter_task.created_at.isoformat() if filter_task.created_at else None,
                        },
                    })

                parent_version_id = filtered_version.parent_version_id
                while parent_version_id:
                    pv_stmt = select(DatasetVersion).where(DatasetVersion.id == parent_version_id)
                    pv_result = await session.execute(pv_stmt)
                    pv = pv_result.scalar_one_or_none()
                    if not pv:
                        break
                    chain.append({
                        "node_type": f"{pv.version_type}_version",
                        "id": pv.id,
                        "name": pv.version_name,
                        "info": {
                            "version_type": pv.version_type,
                            "total_samples": pv.total_samples,
                            "class_distribution": pv.class_distribution or {},
                            "parent_version_id": pv.parent_version_id,
                            "created_at": pv.created_at.isoformat() if pv.created_at else None,
                        },
                    })
                    parent_version_id = pv.parent_version_id

                dataset_stmt = select(Dataset).where(Dataset.id == annotated_version.dataset_id)
                dataset_result = await session.execute(dataset_stmt)
                dataset = dataset_result.scalar_one_or_none()
                if dataset:
                    chain.append({
                        "node_type": "dataset",
                        "id": dataset.id,
                        "name": dataset.name,
                        "info": {
                            "description": dataset.description,
                            "num_classes": dataset.num_classes,
                            "total_samples": dataset.total_samples,
                            "class_distribution": dataset.class_distribution or {},
                            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                        },
                    })

    return {"task_id": task_id, "chain": chain}
