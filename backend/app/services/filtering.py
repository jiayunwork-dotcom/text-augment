import logging
from typing import Optional
from collections import Counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    FilterTask, DatasetVersion, Sample, TaskStatus, SampleSource, SplitType
)
from ..config import FILTER_PRESETS, DEFAULT_PPL_THRESHOLD_MULTIPLIER

logger = logging.getLogger(__name__)

_ppl_model = None
_embedding_model = None
_classifier_model = None


def get_ppl_model():
    global _ppl_model
    if _ppl_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            _ppl_model = (tokenizer, model)
        except Exception as e:
            logger.warning(f"Failed to load PPL model: {e}")
    return _ppl_model


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
    return _embedding_model


def compute_perplexity(text: str, tokenizer, model) -> float:
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def compute_perplexity_batch(texts: list[str], tokenizer, model) -> list[float]:
    import torch
    results = []
    for text in texts:
        try:
            ppl = compute_perplexity(text, tokenizer, model)
            results.append(ppl)
        except Exception:
            results.append(float("inf"))
    return results


def compute_similarity(text_a: str, text_b: str, model) -> float:
    emb_a = model.encode([text_a])
    emb_b = model.encode([text_b])
    from numpy.linalg import norm
    sim = np.dot(emb_a[0], emb_b[0]) / (norm(emb_a[0]) * norm(emb_b[0]) + 1e-8)
    return float(sim)


def compute_similarity_batch(pairs: list[tuple[str, str]], model) -> list[float]:
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    emb_a = model.encode(texts_a)
    emb_b = model.encode(texts_b)
    from numpy.linalg import norm
    results = []
    for i in range(len(pairs)):
        sim = np.dot(emb_a[i], emb_b[i]) / (norm(emb_a[i]) * norm(emb_b[i]) + 1e-8)
        results.append(float(sim))
    return results


def minhash_dedup(texts: list[str], threshold: float = 0.9) -> list[int]:
    try:
        from datasketch import MinHash, LeanMinHash
    except ImportError:
        logger.warning("datasketch not available, skipping dedup")
        return []

    minhashes = []
    for text in texts:
        mh = MinHash(num_perm=128)
        tokens = text.split()
        for token in tokens:
            mh.update(token.encode("utf-8"))
        minhashes.append(mh)

    duplicates = set()
    for i in range(len(minhashes)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(minhashes)):
            if j in duplicates:
                continue
            if minhashes[i].jaccard(minhashes[j]) > threshold:
                duplicates.add(j)

    return list(duplicates)


async def execute_filter_task(
    session: AsyncSession,
    task_id: int,
) -> None:
    stmt = select(FilterTask).where(FilterTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        return

    task.status = TaskStatus.running
    await session.commit()

    try:
        preset = FILTER_PRESETS.get(task.strictness.value, FILTER_PRESETS["standard"])
        ppl_multiplier = preset["ppl_multiplier"]
        similarity_threshold = preset["similarity_threshold"]
        jaccard_threshold = preset["jaccard_threshold"]
        label_confidence_threshold = preset["label_confidence_threshold"]

        stmt = select(Sample).where(
            Sample.version_id == task.version_id,
            Sample.is_filtered == False,
        )
        result = await session.execute(stmt)
        all_samples = result.scalars().all()

        original_samples = [s for s in all_samples if s.source == SampleSource.original]
        augmented_samples = [s for s in all_samples if s.source != SampleSource.original]

        task.total_samples = len(augmented_samples)
        await session.commit()

        if not augmented_samples:
            task.status = TaskStatus.completed
            task.passed_samples = 0
            task.filtered_samples = 0
            task.completed_at = __import__("datetime").datetime.utcnow()
            await session.commit()
            return

        ppl_model_data = get_ppl_model()
        original_ppls = []
        if ppl_model_data:
            tokenizer, model = ppl_model_data
            original_texts = [s.text for s in original_samples]
            original_ppls = compute_perplexity_batch(original_texts, tokenizer, model)
            avg_original_ppl = np.mean(original_ppls) if original_ppls else 10.0
            ppl_threshold = avg_original_ppl * ppl_multiplier

            aug_texts = [s.text for s in augmented_samples]
            aug_ppls = compute_perplexity_batch(aug_texts, tokenizer, model)

            for i, sample in enumerate(augmented_samples):
                sample.perplexity = aug_ppls[i]
                if aug_ppls[i] > ppl_threshold:
                    sample.is_filtered = True
                    sample.filter_reason = "high_ppl"

        task.ppl_filtered = sum(1 for s in augmented_samples if s.is_filtered and s.filter_reason == "high_ppl")
        await session.commit()

        remaining_aug = [s for s in augmented_samples if not s.is_filtered]

        embedding_model = get_embedding_model()
        if embedding_model and original_samples:
            pairs = []
            for aug_s in remaining_aug:
                source_stmt = select(Sample).where(Sample.id == aug_s.source_sample_id)
                source_result = await session.execute(source_stmt)
                source_s = source_result.scalar_one_or_none()
                if source_s:
                    pairs.append((aug_s.text, source_s.text))
                else:
                    pairs.append((aug_s.text, original_samples[0].text))

            if pairs:
                similarities = compute_similarity_batch(pairs, embedding_model)
                for i, sample in enumerate(remaining_aug):
                    sample.similarity_score = similarities[i]
                    if similarities[i] < similarity_threshold:
                        sample.is_filtered = True
                        sample.filter_reason = sample.filter_reason or "low_similarity"

        task.similarity_filtered = sum(
            1 for s in augmented_samples if s.is_filtered and "similarity" in (s.filter_reason or "")
        )
        await session.commit()

        remaining_aug = [s for s in augmented_samples if not s.is_filtered]

        if remaining_aug:
            dup_texts = [s.text for s in remaining_aug]
            original_texts_set = [s.text for s in original_samples]
            all_texts = original_texts_set + dup_texts
            dup_indices = minhash_dedup(all_texts, jaccard_threshold)
            dup_indices_in_aug = [idx - len(original_texts_set) for idx in dup_indices if idx >= len(original_texts_set)]
            for idx in dup_indices_in_aug:
                if 0 <= idx < len(remaining_aug):
                    remaining_aug[idx].is_filtered = True
                    remaining_aug[idx].filter_reason = remaining_aug[idx].filter_reason or "duplicate"

        task.dedup_filtered = sum(
            1 for s in augmented_samples if s.is_filtered and "duplicate" in (s.filter_reason or "")
        )
        await session.commit()

        task.label_filtered = 0
        task.passed_samples = sum(1 for s in augmented_samples if not s.is_filtered)
        task.filtered_samples = sum(1 for s in augmented_samples if s.is_filtered)
        task.status = TaskStatus.completed
        task.completed_at = __import__("datetime").datetime.utcnow()

        version_stmt = select(DatasetVersion).where(DatasetVersion.id == task.version_id)
        version_result = await session.execute(version_stmt)
        version = version_result.scalar_one_or_none()
        if version:
            passed_count = sum(1 for s in all_samples if not s.is_filtered)
            class_dist = Counter()
            for s in all_samples:
                if not s.is_filtered:
                    class_dist[s.label] += 1
            version.total_samples = passed_count
            version.class_distribution = dict(class_dist)

        await session.commit()

    except Exception as e:
        logger.exception(f"Filter task {task_id} failed")
        task.status = TaskStatus.failed
        task.error_message = str(e)
        await session.commit()


async def create_filtered_version(session: AsyncSession, filter_task: FilterTask) -> DatasetVersion:
    stmt = select(DatasetVersion).where(DatasetVersion.id == filter_task.version_id)
    result = await session.execute(stmt)
    source_version = result.scalar_one_or_none()
    if not source_version:
        raise ValueError("Source version not found")

    filtered_version = DatasetVersion(
        dataset_id=source_version.dataset_id,
        version_name=f"filtered_{filter_task.strictness.value}",
        version_type="filtered",
        parent_version_id=source_version.id,
        filter_strictness=filter_task.strictness.value,
    )
    session.add(filtered_version)
    await session.flush()

    stmt = select(Sample).where(
        Sample.version_id == filter_task.version_id,
        Sample.is_filtered == False,
    )
    result = await session.execute(stmt)
    passed_samples = result.scalars().all()

    class_dist = Counter()
    for s in passed_samples:
        new_sample = Sample(
            version_id=filtered_version.id,
            text=s.text,
            label=s.label,
            split=s.split,
            source=s.source,
            source_sample_id=s.source_sample_id,
            perplexity=s.perplexity,
            similarity_score=s.similarity_score,
            label_confidence=s.label_confidence,
        )
        session.add(new_sample)
        class_dist[s.label] += 1

    filtered_version.total_samples = len(passed_samples)
    filtered_version.class_distribution = dict(class_dist)
    filtered_version.split_ratios = source_version.split_ratios

    filter_task.target_version_id = filtered_version.id
    await session.commit()

    return filtered_version
