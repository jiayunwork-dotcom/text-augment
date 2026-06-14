import logging
import csv
import io
import asyncio
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

import httpx
from sqlalchemy import select, and_, or_, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    AnnotationQueue, AnnotationItem, AnnotationRecord,
    QueueStatus, AnnotationStatus, AnnotationDecision,
    DatasetVersion, Sample, Dataset, PriorityStrategy,
    WebhookLog, RecommendedFilterConfig,
)
from ..models.schemas import (
    AnnotationQueueCreate, QueueProgressStats,
    ConsistencyReport, AnnotatorStats,
    AnnotatorPerformanceResponse, BulkImportResult,
)
from ..config import FILTER_PRESETS

logger = logging.getLogger(__name__)


def compute_uncertainty_score(sample: Sample) -> float:
    scores = []

    if sample.label_confidence is not None:
        if sample.label_confidence < 0.5:
            scores.append(1.0)
        elif sample.label_confidence <= 0.8:
            scores.append(1.0 - abs(0.65 - sample.label_confidence) / 0.3)
        else:
            scores.append(max(0.1, 1.0 - (sample.label_confidence - 0.8) / 0.2))
    else:
        if sample.source and sample.source.value != "original":
            scores.append(0.6)
        else:
            scores.append(0.3)

    if sample.is_filtered and sample.filter_reason:
        scores.append(0.8)

    if sample.similarity_score is not None:
        if sample.similarity_score < 0.7:
            scores.append(0.8)
        else:
            scores.append(max(0.1, (0.85 - sample.similarity_score) / 0.3))
    elif sample.source and sample.source.value != "original":
        scores.append(0.4)

    if sample.perplexity is not None:
        if sample.perplexity > 50:
            scores.append(0.8)
        else:
            scores.append(min(0.5, sample.perplexity / 100.0))

    if not scores:
        return 0.3

    return round(sum(scores) / len(scores), 4)


async def check_version_for_annotation(
    session: AsyncSession,
    version_id: int,
) -> dict:
    stmt = select(DatasetVersion).where(DatasetVersion.id == version_id)
    result = await session.execute(stmt)
    version = result.scalar_one_or_none()

    if not version:
        return {"valid": False, "reason": "Version not found"}

    if version.version_type != "filtered":
        return {
            "valid": False,
            "reason": f"Only 'filtered' versions can create annotation queues. "
                      f"Current version type is '{version.version_type}'.",
            "version_type": version.version_type,
        }

    stmt2 = (
        select(AnnotationQueue)
        .where(
            and_(
                AnnotationQueue.version_id == version_id,
                AnnotationQueue.status.in_([
                    QueueStatus.pending,
                    QueueStatus.in_progress,
                    QueueStatus.completed,
                ]),
            )
        )
    )
    result2 = await session.execute(stmt2)
    active_queues = result2.scalars().all()

    if active_queues:
        return {
            "valid": False,
            "reason": f"Version already has an active annotation queue (ID: {active_queues[0].id}). "
                      f"Please close it first before creating a new one.",
            "active_queue_id": active_queues[0].id,
        }

    return {"valid": True, "version": version}


async def select_uncertain_samples(
    session: AsyncSession,
    version_id: int,
    capacity: int,
) -> list[tuple[Sample, float]]:
    stmt = select(Sample).where(Sample.version_id == version_id)
    result = await session.execute(stmt)
    all_samples = result.scalars().all()

    scored = []
    for s in all_samples:
        score = compute_uncertainty_score(s)
        scored.append((s, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:capacity]


async def create_annotation_queue(
    session: AsyncSession,
    request: AnnotationQueueCreate,
) -> AnnotationQueue:
    validation = await check_version_for_annotation(session, request.version_id)
    if not validation["valid"]:
        raise ValueError(validation["reason"])

    version = validation["version"]

    if request.review_mode == "multi":
        if request.num_reviewers % 2 == 0:
            raise ValueError(
                f"Multi-review mode requires odd number of reviewers for majority voting. "
                f"Got {request.num_reviewers}, please use an odd number."
            )

    if request.webhook_thresholds:
        for t in request.webhook_thresholds:
            if t <= 0 or t > 100:
                raise ValueError(f"Webhook threshold {t} is invalid. Must be between 0 and 100.")

    name = request.name or f"annotation_queue_v{version.id}"

    queue = AnnotationQueue(
        version_id=request.version_id,
        name=name,
        status=QueueStatus.pending,
        capacity=request.capacity,
        review_mode=request.review_mode,
        num_reviewers=request.num_reviewers,
        lock_timeout_minutes=request.lock_timeout_minutes,
        created_by=request.created_by or None,
        priority_strategy=request.priority_strategy,
        webhook_url=request.webhook_url,
        webhook_thresholds=sorted(set(request.webhook_thresholds)),
        triggered_thresholds=[],
    )
    session.add(queue)
    await session.flush()

    selected = await select_uncertain_samples(session, request.version_id, request.capacity)

    for sample, score in selected:
        item = AnnotationItem(
            queue_id=queue.id,
            sample_id=sample.id,
            uncertainty_score=round(score, 4),
            status=AnnotationStatus.pending,
        )
        session.add(item)

    queue.capacity = len(selected)
    if len(selected) == 0:
        queue.status = QueueStatus.completed
        queue.completed_at = datetime.utcnow()

    await session.commit()
    await session.refresh(queue)

    return queue


async def _get_annotated_class_distribution(
    session: AsyncSession,
    queue_id: int,
) -> dict[str, int]:
    stmt = (
        select(Sample.label, func.count(AnnotationItem.id))
        .select_from(AnnotationItem)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status.in_([
                    AnnotationStatus.annotated,
                    AnnotationStatus.arbitrated,
                ]),
            )
        )
        .group_by(Sample.label)
    )
    result = await session.execute(stmt)
    rows = result.all()
    return {label: count for label, count in rows}


async def _get_all_classes_in_queue(
    session: AsyncSession,
    queue_id: int,
) -> list[str]:
    stmt = (
        select(Sample.label)
        .select_from(AnnotationItem)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(AnnotationItem.queue_id == queue_id)
        .distinct()
    )
    result = await session.execute(stmt)
    return [row[0] for row in result.all()]


async def _compute_class_rarity_scores(
    session: AsyncSession,
    queue_id: int,
    items: list[tuple[AnnotationItem, Sample]],
) -> dict[int, float]:
    annotated_dist = await _get_annotated_class_distribution(session, queue_id)
    all_classes = await _get_all_classes_in_queue(session, queue_id)

    for cls in all_classes:
        if cls not in annotated_dist:
            annotated_dist[cls] = 0

    if not annotated_dist:
        return {item.id: 0.5 for item, _ in items}

    max_count = max(annotated_dist.values()) if annotated_dist else 1
    if max_count == 0:
        max_count = 1

    rarity_scores = {}
    for item, sample in items:
        cls_count = annotated_dist.get(sample.label, 0)
        rarity = 1.0 - (cls_count / max_count)
        rarity_scores[item.id] = max(0.0, min(1.0, rarity))

    return rarity_scores


async def _compute_priority_order(
    session: AsyncSession,
    queue: AnnotationQueue,
    rows: list[tuple[AnnotationItem, Sample]],
) -> list[tuple[AnnotationItem, Sample]]:
    if queue.priority_strategy == PriorityStrategy.uncertainty or not rows:
        return rows

    item_sample_map = {item.id: (item, sample) for item, sample in rows}

    if queue.priority_strategy == PriorityStrategy.class_balance:
        rarity_scores = await _compute_class_rarity_scores(session, queue.id, rows)
        sorted_items = sorted(
            rows,
            key=lambda x: (
                rarity_scores.get(x[0].id, 0.0),
                x[0].uncertainty_score,
            ),
            reverse=True,
        )
        return sorted_items

    elif queue.priority_strategy == PriorityStrategy.hybrid:
        rarity_scores = await _compute_class_rarity_scores(session, queue.id, rows)
        uncertainty_weight = 0.7
        rarity_weight = 0.3

        scored_rows = []
        for item, sample in rows:
            uncertainty_norm = item.uncertainty_score
            rarity = rarity_scores.get(item.id, 0.0)
            combined = (
                uncertainty_weight * uncertainty_norm
                + rarity_weight * rarity
            )
            scored_rows.append((item, sample, combined))

        scored_rows.sort(key=lambda x: x[2], reverse=True)
        return [(item, sample) for item, sample, _ in scored_rows]

    return rows


def _is_lock_expired(item: AnnotationItem, timeout_minutes: int) -> bool:
    if item.locked_at is None:
        return True
    cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
    return item.locked_at < cutoff


async def release_expired_locks(session: AsyncSession, queue: AnnotationQueue):
    stmt = (
        select(AnnotationItem)
        .where(
            and_(
                AnnotationItem.queue_id == queue.id,
                AnnotationItem.status == AnnotationStatus.locked,
            )
        )
    )
    result = await session.execute(stmt)
    locked_items = result.scalars().all()

    for item in locked_items:
        if _is_lock_expired(item, queue.lock_timeout_minutes):
            item.status = AnnotationStatus.pending
            item.locked_by = None
            item.locked_at = None

    await session.commit()


async def claim_tasks(
    session: AsyncSession,
    queue_id: int,
    annotator_id: str,
    batch_size: int,
) -> list[dict]:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    if queue.status in [QueueStatus.applied, QueueStatus.closed]:
        raise ValueError(f"Queue is {queue.status.value}, cannot claim tasks")

    if queue.status == QueueStatus.pending:
        queue.status = QueueStatus.in_progress
        queue.started_at = datetime.utcnow()

    await release_expired_locks(session, queue)

    source_ids_stmt = (
        select(Sample.id, Sample.source_sample_id)
        .select_from(AnnotationItem)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status == AnnotationStatus.annotated,
                AnnotationRecord.annotator_id == annotator_id,
            )
        )
        .join(AnnotationRecord, AnnotationItem.id == AnnotationRecord.item_id)
    )
    source_result = await session.execute(source_ids_stmt)
    source_ids_rows = source_result.all()
    augmented_ids_created_by_annotator = set()
    for sid, source_sid in source_ids_rows:
        if source_sid:
            augmented_ids_created_by_annotator.add(source_sid)

    stmt2 = (
        select(AnnotationItem, Sample)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status.in_([AnnotationStatus.pending, AnnotationStatus.locked]),
            )
        )
        .order_by(AnnotationItem.uncertainty_score.desc())
    )
    result2 = await session.execute(stmt2)
    rows = result2.all()

    rows = await _compute_priority_order(session, queue, list(rows))

    claimed = []
    for item, sample in rows:
        if len(claimed) >= batch_size:
            break

        if item.status == AnnotationStatus.locked and item.locked_by != annotator_id:
            continue

        existing_record_stmt = (
            select(AnnotationRecord)
            .where(
                and_(
                    AnnotationRecord.item_id == item.id,
                    AnnotationRecord.annotator_id == annotator_id,
                )
            )
        )
        existing_record = await session.execute(existing_record_stmt)
        if existing_record.scalar_one_or_none():
            continue

        if sample.source_sample_id and sample.source_sample_id in augmented_ids_created_by_annotator:
            continue

        current_records_stmt = (
            select(func.count(AnnotationRecord.id))
            .where(AnnotationRecord.item_id == item.id)
        )
        current_records_count = await session.execute(current_records_stmt)
        num_records = current_records_count.scalar() or 0
        if num_records >= queue.num_reviewers and item.status == AnnotationStatus.annotated:
            continue

        item.status = AnnotationStatus.locked
        item.locked_by = annotator_id
        item.locked_at = datetime.utcnow()

        claimed.append({
            "item_id": item.id,
            "sample_id": sample.id,
            "text": sample.text,
            "current_label": sample.label,
            "predicted_label": None,
            "confidence": sample.label_confidence,
            "similarity_score": sample.similarity_score,
            "perplexity": sample.perplexity,
            "uncertainty_score": item.uncertainty_score,
            "source_sample_id": sample.source_sample_id,
        })

    await session.commit()

    if len(claimed) < batch_size:
        logger.info(f"Annotator {annotator_id} claimed {len(claimed)} tasks, requested {batch_size}")

    return claimed


async def _resolve_item_status(
    session: AsyncSession,
    item: AnnotationItem,
    queue: AnnotationQueue,
) -> AnnotationItem:
    stmt = (
        select(AnnotationRecord)
        .where(AnnotationRecord.item_id == item.id)
    )
    result = await session.execute(stmt)
    records = result.scalars().all()

    if len(records) < queue.num_reviewers:
        if len(records) > 0 and item.status != AnnotationStatus.locked:
            item.status = AnnotationStatus.pending
        return item

    decisions = [r.decision.value for r in records]
    decision_counts = Counter(decisions)

    majority_count = (queue.num_reviewers // 2) + 1
    has_majority = any(c >= majority_count for c in decision_counts.values())

    if has_majority:
        majority_decision = decision_counts.most_common(1)[0][0]
        item.final_decision = AnnotationDecision(majority_decision)

        if item.final_decision == AnnotationDecision.relabel:
            relabel_records = [r for r in records if r.decision == AnnotationDecision.relabel]
            label_counts = Counter(r.new_label for r in relabel_records if r.new_label)
            if label_counts:
                item.final_label = label_counts.most_common(1)[0][0]

        for record in records:
            record.is_final_decision = (
                record.decision.value == majority_decision
            )

        item.status = AnnotationStatus.annotated
    else:
        item.status = AnnotationStatus.disputed

    return item


async def submit_annotations(
    session: AsyncSession,
    queue_id: int,
    annotator_id: str,
    items: list,
) -> dict:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    if queue.status in [QueueStatus.applied, QueueStatus.closed]:
        raise ValueError(f"Queue is {queue.status.value}, cannot submit annotations")

    processed_count = 0
    errors = []

    for submit_item in items:
        try:
            stmt2 = select(AnnotationItem).where(AnnotationItem.id == submit_item.item_id)
            result2 = await session.execute(stmt2)
            item = result2.scalar_one_or_none()

            if not item:
                errors.append({"item_id": submit_item.item_id, "error": "Item not found"})
                continue

            if item.queue_id != queue_id:
                errors.append({"item_id": submit_item.item_id, "error": "Item does not belong to this queue"})
                continue

            if item.locked_by != annotator_id:
                if item.status == AnnotationStatus.locked:
                    errors.append({"item_id": submit_item.item_id, "error": "Item locked by another annotator"})
                    continue

            stmt3 = (
                select(Sample)
                .join(AnnotationItem, Sample.id == AnnotationItem.sample_id)
                .where(AnnotationItem.id == submit_item.item_id)
            )
            result3 = await session.execute(stmt3)
            sample = result3.scalar_one_or_none()

            if sample and sample.source_sample_id:
                pass

            if submit_item.decision == AnnotationDecision.relabel:
                if not submit_item.new_label:
                    errors.append({"item_id": submit_item.item_id, "error": "new_label is required for relabel decision"})
                    continue

            existing_stmt = (
                select(AnnotationRecord)
                .where(
                    and_(
                        AnnotationRecord.item_id == item.id,
                        AnnotationRecord.annotator_id == annotator_id,
                    )
                )
            )
            existing_result = await session.execute(existing_stmt)
            existing_record = existing_result.scalar_one_or_none()

            if existing_record:
                existing_record.decision = submit_item.decision
                existing_record.new_label = submit_item.new_label
                existing_record.comment = submit_item.comment
                existing_record.submitted_at = datetime.utcnow()
                if item.locked_at:
                    duration = (datetime.utcnow() - item.locked_at).total_seconds()
                    existing_record.annotation_duration_seconds = duration
            else:
                now = datetime.utcnow()
                duration = None
                if item.locked_at:
                    duration = (now - item.locked_at).total_seconds()
                record = AnnotationRecord(
                    item_id=item.id,
                    annotator_id=annotator_id,
                    decision=submit_item.decision,
                    new_label=submit_item.new_label,
                    comment=submit_item.comment,
                    locked_at=item.locked_at,
                    submitted_at=now,
                    annotation_duration_seconds=duration,
                )
                session.add(record)

            item.locked_by = None
            item.locked_at = None

            await session.flush()
            await _resolve_item_status(session, item, queue)

            processed_count += 1
        except Exception as e:
            logger.exception(f"Error processing item {getattr(submit_item, 'item_id', '?')}")
            errors.append({"item_id": getattr(submit_item, "item_id", "?"), "error": str(e)})

    await session.commit()

    all_items_stmt = (
        select(func.count(AnnotationItem.id))
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status.in_([AnnotationStatus.pending, AnnotationStatus.locked]),
            )
        )
    )
    all_items_result = await session.execute(all_items_stmt)
    remaining = all_items_result.scalar() or 0

    if remaining == 0 and queue.status in [QueueStatus.pending, QueueStatus.in_progress]:
        queue.status = QueueStatus.completed
        queue.completed_at = datetime.utcnow()
        await session.commit()

    await _check_and_trigger_webhooks(session, queue)

    return {
        "processed_count": processed_count,
        "errors": errors,
        "remaining_pending": remaining,
    }


async def release_locks(
    session: AsyncSession,
    queue_id: int,
    annotator_id: str,
    item_ids: list[int],
) -> dict:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    query = (
        select(AnnotationItem)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.locked_by == annotator_id,
            )
        )
    )
    if item_ids:
        query = query.where(AnnotationItem.id.in_(item_ids))

    result2 = await session.execute(query)
    items = result2.scalars().all()

    released = 0
    for item in items:
        item.status = AnnotationStatus.pending
        item.locked_by = None
        item.locked_at = None
        released += 1

    await session.commit()
    return {"released_count": released}


async def get_queue_progress(
    session: AsyncSession,
    queue_id: int,
) -> QueueProgressStats:
    items_stmt = select(AnnotationItem).where(AnnotationItem.queue_id == queue_id)
    result = await session.execute(items_stmt)
    items = result.scalars().all()

    stats = QueueProgressStats(total=len(items))

    for item in items:
        if item.status == AnnotationStatus.pending:
            stats.pending += 1
        elif item.status == AnnotationStatus.locked:
            stats.locked += 1
        elif item.status == AnnotationStatus.annotated:
            stats.annotated += 1
            if item.final_decision == AnnotationDecision.confirm:
                stats.confirm_count += 1
            elif item.final_decision == AnnotationDecision.relabel:
                stats.relabel_count += 1
            elif item.final_decision == AnnotationDecision.discard:
                stats.discard_count += 1
        elif item.status == AnnotationStatus.disputed:
            stats.disputed += 1
        elif item.status == AnnotationStatus.arbitrated:
            stats.arbitrated += 1
            if item.final_decision == AnnotationDecision.confirm:
                stats.confirm_count += 1
            elif item.final_decision == AnnotationDecision.relabel:
                stats.relabel_count += 1
            elif item.final_decision == AnnotationDecision.discard:
                stats.discard_count += 1

    finalized = stats.annotated + stats.arbitrated
    if finalized > 0:
        stats.confirm_rate = round(stats.confirm_count / finalized, 4)
        stats.relabel_rate = round(stats.relabel_count / finalized, 4)
        stats.discard_rate = round(stats.discard_count / finalized, 4)

    return stats


async def get_disputed_items(
    session: AsyncSession,
    queue_id: int,
) -> list[dict]:
    stmt = (
        select(AnnotationItem, Sample)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status == AnnotationStatus.disputed,
            )
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    response = []
    for item, sample in rows:
        records_stmt = (
            select(AnnotationRecord)
            .where(AnnotationRecord.item_id == item.id)
        )
        records_result = await session.execute(records_stmt)
        records = records_result.scalars().all()

        records_data = [
            {
                "annotator_id": r.annotator_id,
                "decision": r.decision.value,
                "new_label": r.new_label,
                "comment": r.comment,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]

        response.append({
            "item_id": item.id,
            "sample_id": sample.id,
            "text": sample.text,
            "current_label": sample.label,
            "records": records_data,
            "uncertainty_score": item.uncertainty_score,
        })

    return response


async def arbitrate_items(
    session: AsyncSession,
    queue_id: int,
    arbitrator_id: str,
    items: list,
) -> dict:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    processed_count = 0
    errors = []

    for arb_item in items:
        try:
            stmt2 = select(AnnotationItem).where(AnnotationItem.id == arb_item.item_id)
            result2 = await session.execute(stmt2)
            item = result2.scalar_one_or_none()

            if not item:
                errors.append({"item_id": arb_item.item_id, "error": "Item not found"})
                continue

            if item.status != AnnotationStatus.disputed:
                errors.append({"item_id": arb_item.item_id, "error": f"Item status is {item.status.value}, not disputed"})
                continue

            if arb_item.decision == AnnotationDecision.relabel and not arb_item.new_label:
                errors.append({"item_id": arb_item.item_id, "error": "new_label required for relabel"})
                continue

            item.final_decision = arb_item.decision
            item.final_label = arb_item.new_label
            item.status = AnnotationStatus.arbitrated
            item.arbitrated_by = arbitrator_id
            item.arbitrated_at = datetime.utcnow()

            records_stmt = select(AnnotationRecord).where(AnnotationRecord.item_id == item.id)
            records_result = await session.execute(records_stmt)
            item_records = records_result.scalars().all()
            for rec in item_records:
                rec.is_final_decision = (rec.decision == arb_item.decision)

            processed_count += 1
        except Exception as e:
            errors.append({"item_id": getattr(arb_item, "item_id", "?"), "error": str(e)})

    await session.commit()

    remaining_stmt = (
        select(func.count(AnnotationItem.id))
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status.in_([
                    AnnotationStatus.pending,
                    AnnotationStatus.locked,
                    AnnotationStatus.disputed,
                ]),
            )
        )
    )
    remaining_result = await session.execute(remaining_stmt)
    remaining = remaining_result.scalar() or 0

    if remaining == 0 and queue.status in [QueueStatus.pending, QueueStatus.in_progress, QueueStatus.completed]:
        queue.status = QueueStatus.completed
        queue.completed_at = datetime.utcnow()
        await session.commit()

    await _check_and_trigger_webhooks(session, queue)

    return {"processed_count": processed_count, "errors": errors, "remaining": remaining}


async def apply_queue_results(
    session: AsyncSession,
    queue_id: int,
    applied_by: str = "",
) -> DatasetVersion:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    if queue.status == QueueStatus.applied:
        raise ValueError(f"Queue already applied. Target version ID: {queue.target_version_id}")

    progress = await get_queue_progress(session, queue_id)
    remaining = progress.pending + progress.locked + progress.disputed

    if remaining > 0:
        raise ValueError(
            f"Queue not fully completed. Remaining unprocessed: {remaining} "
            f"(pending: {progress.pending}, locked: {progress.locked}, disputed: {progress.disputed})"
        )

    if queue.status not in (QueueStatus.completed, QueueStatus.closed):
        queue.status = QueueStatus.completed
        queue.completed_at = queue.completed_at or datetime.utcnow()

    version_stmt = select(DatasetVersion).where(DatasetVersion.id == queue.version_id)
    version_result = await session.execute(version_stmt)
    source_version = version_result.scalar_one_or_none()

    if not source_version:
        raise ValueError("Source version not found")

    items_stmt = (
        select(AnnotationItem, Sample)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(AnnotationItem.queue_id == queue_id)
    )
    items_result = await session.execute(items_stmt)
    annotated_rows = items_result.all()

    annotated_samples = {item.sample_id: (item, sample) for item, sample in annotated_rows}

    all_samples_stmt = select(Sample).where(Sample.version_id == queue.version_id)
    all_result = await session.execute(all_samples_stmt)
    all_samples = all_result.scalars().all()

    new_version = DatasetVersion(
        dataset_id=source_version.dataset_id,
        version_name=f"annotated_{queue.name}",
        version_type="annotated",
        parent_version_id=source_version.id,
        filter_strictness=source_version.filter_strictness,
    )
    session.add(new_version)
    await session.flush()

    class_dist = Counter()
    kept_count = 0

    for sample in all_samples:
        if sample.id in annotated_samples:
            item, _ = annotated_samples[sample.id]

            if item.final_decision == AnnotationDecision.discard:
                continue

            new_label = sample.label
            if item.final_decision == AnnotationDecision.relabel and item.final_label:
                new_label = item.final_label

            new_sample = Sample(
                version_id=new_version.id,
                text=sample.text,
                label=new_label,
                split=sample.split,
                source=sample.source,
                source_sample_id=sample.source_sample_id,
                perplexity=sample.perplexity,
                similarity_score=sample.similarity_score,
                label_confidence=sample.label_confidence,
                is_manually_approved=True,
            )
            session.add(new_sample)
            class_dist[new_label] += 1
            kept_count += 1
        else:
            new_sample = Sample(
                version_id=new_version.id,
                text=sample.text,
                label=sample.label,
                split=sample.split,
                source=sample.source,
                source_sample_id=sample.source_sample_id,
                perplexity=sample.perplexity,
                similarity_score=sample.similarity_score,
                label_confidence=sample.label_confidence,
            )
            session.add(new_sample)
            class_dist[sample.label] += 1
            kept_count += 1

    new_version.total_samples = kept_count
    new_version.class_distribution = dict(class_dist)
    new_version.split_ratios = source_version.split_ratios

    queue.status = QueueStatus.applied
    queue.applied_at = datetime.utcnow()
    queue.target_version_id = new_version.id

    await session.commit()
    await session.refresh(new_version)

    await _generate_recommended_filter_config(session, queue, new_version)

    return new_version


FILTER_REASON_TO_PARAM = {
    "high_ppl": "ppl_multiplier",
    "low_similarity": "similarity_threshold",
    "duplicate": "jaccard_threshold",
    "label": "label_confidence_threshold",
}


async def _analyze_filter_reason_stats(
    session: AsyncSession,
    queue_id: int,
    augmented_version_id: Optional[int] = None,
) -> dict[str, dict]:
    logger.info(f"[FilterStats] Starting analysis for queue {queue_id}")

    if augmented_version_id is None:
        queue_stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
        queue_result = await session.execute(queue_stmt)
        queue = queue_result.scalar_one_or_none()
        if not queue:
            logger.warning(f"[FilterStats] Queue {queue_id} not found")
            return {}

        source_version_stmt = select(DatasetVersion).where(
            DatasetVersion.id == queue.version_id
        )
        source_version_result = await session.execute(source_version_stmt)
        source_version = source_version_result.scalar_one_or_none()
        if source_version and source_version.parent_version_id:
            augmented_version_id = source_version.parent_version_id
            logger.info(f"[FilterStats] Using augmented version_id={augmented_version_id} "
                        f"(from source version parent)")
        elif source_version:
            augmented_version_id = queue.version_id
            logger.info(f"[FilterStats] No parent version, using queue's source version_id={augmented_version_id} as fallback")

    if augmented_version_id is None:
        logger.warning(f"[FilterStats] Cannot determine version for queue {queue_id}")
        return {}

    finalized_stmt = (
        select(AnnotationItem, Sample)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                AnnotationItem.status.in_([
                    AnnotationStatus.annotated,
                    AnnotationStatus.arbitrated,
                ]),
            )
        )
    )
    finalized_result = await session.execute(finalized_stmt)
    finalized_items = finalized_result.all()

    logger.info(f"[FilterStats] Found {len(finalized_items)} finalized annotated items")

    if not finalized_items:
        return {}

    filtered_sample_to_source: dict[int, AnnotationItem] = {}
    aug_sample_ids = []

    for item, filtered_sample in finalized_items:
        aug_sample_id = filtered_sample.source_sample_id
        if aug_sample_id is None:
            aug_sample_id = filtered_sample.id
            logger.debug(f"[FilterStats] sample {filtered_sample.id} has no source_sample_id, "
                        f"using own id={aug_sample_id}")

        aug_sample_ids.append(aug_sample_id)
        filtered_sample_to_source[aug_sample_id] = item

    logger.info(f"[FilterStats] Looking up {len(aug_sample_ids)} samples in version {augmented_version_id} "
                f"with is_filtered=True and filter_reason IS NOT NULL")

    aug_samples_stmt = select(Sample).where(
        and_(
            Sample.id.in_(aug_sample_ids),
            Sample.version_id == augmented_version_id,
        )
    )
    aug_samples_result = await session.execute(aug_samples_stmt)
    all_aug_samples = aug_samples_result.scalars().all()

    logger.info(f"[FilterStats] Found {len(all_aug_samples)} samples in aug version "
                f"(is_filtered=True and filter_reason: "
                f"{sum(1 for s in all_aug_samples if s.is_filtered and s.filter_reason)})")

    filter_reason_counts = defaultdict(lambda: {
        "total": 0, "confirm_count": 0, "discard_count": 0, "relabel_count": 0
    })

    matched_count = 0
    for aug_sample in all_aug_samples:
        if not aug_sample.is_filtered or not aug_sample.filter_reason:
            continue

        item = filtered_sample_to_source.get(aug_sample.id)
        if not item:
            continue

        matched_count += 1
        reason = aug_sample.filter_reason
        filter_reason_counts[reason]["total"] += 1
        if item.final_decision == AnnotationDecision.confirm:
            filter_reason_counts[reason]["confirm_count"] += 1
        elif item.final_decision == AnnotationDecision.discard:
            filter_reason_counts[reason]["discard_count"] += 1
        elif item.final_decision == AnnotationDecision.relabel:
            filter_reason_counts[reason]["relabel_count"] += 1

    logger.info(f"[FilterStats] Matched {matched_count} samples with filter reasons from aug version")

    if matched_count == 0:
        logger.info(f"[FilterStats] Trying fallback: checking filter_reason in queue source version samples")
        for item, filtered_sample in finalized_items:
            reason = filtered_sample.filter_reason
            if not reason:
                continue
            item_obj = item
            filter_reason_counts[reason]["total"] += 1
            if item_obj.final_decision == AnnotationDecision.confirm:
                filter_reason_counts[reason]["confirm_count"] += 1
            elif item_obj.final_decision == AnnotationDecision.discard:
                filter_reason_counts[reason]["discard_count"] += 1
            elif item_obj.final_decision == AnnotationDecision.relabel:
                filter_reason_counts[reason]["relabel_count"] += 1

        matched_count = sum(c["total"] for c in filter_reason_counts.values())
        logger.info(f"[FilterStats] Fallback found {matched_count} samples with filter reasons")

    stats = {}
    for reason, counts in filter_reason_counts.items():
        total = counts["total"]
        if total == 0:
            continue
        confirm_rate = counts["confirm_count"] / total
        discard_rate = counts["discard_count"] / total
        relabel_rate = counts["relabel_count"] / total
        stats[reason] = {
            "total": total,
            "confirm_count": counts["confirm_count"],
            "discard_count": counts["discard_count"],
            "relabel_count": counts["relabel_count"],
            "confirm_rate": round(confirm_rate, 4),
            "discard_rate": round(discard_rate, 4),
            "relabel_rate": round(relabel_rate, 4),
        }

    logger.info(f"[FilterStats] Final result: {len(stats)} filter reasons with data: {stats}")
    return stats


async def _analyze_filter_reason_stats_fallback(
    session: AsyncSession,
    queue_id: int,
) -> dict[str, dict]:
    items_stmt = (
        select(AnnotationItem, Sample)
        .join(Sample, AnnotationItem.sample_id == Sample.id)
        .where(
            and_(
                AnnotationItem.queue_id == queue_id,
                Sample.filter_reason.isnot(None),
                AnnotationItem.status.in_([
                    AnnotationStatus.annotated,
                    AnnotationStatus.arbitrated,
                ]),
            )
        )
    )
    items_result = await session.execute(items_stmt)
    annotated_items = items_result.all()

    if not annotated_items:
        logger.info(f"No annotated items with filter_reason found for queue {queue_id}")
        return {}

    filter_reason_counts = defaultdict(lambda: {
        "total": 0, "confirm_count": 0, "discard_count": 0, "relabel_count": 0
    })

    for item, sample in annotated_items:
        reason = sample.filter_reason
        if not reason:
            continue

        filter_reason_counts[reason]["total"] += 1
        if item.final_decision == AnnotationDecision.confirm:
            filter_reason_counts[reason]["confirm_count"] += 1
        elif item.final_decision == AnnotationDecision.discard:
            filter_reason_counts[reason]["discard_count"] += 1
        elif item.final_decision == AnnotationDecision.relabel:
            filter_reason_counts[reason]["relabel_count"] += 1

    stats = {}
    for reason, counts in filter_reason_counts.items():
        total = counts["total"]
        if total == 0:
            continue
        confirm_rate = counts["confirm_count"] / total
        discard_rate = counts["discard_count"] / total
        relabel_rate = counts["relabel_count"] / total
        stats[reason] = {
            "total": total,
            "confirm_count": counts["confirm_count"],
            "discard_count": counts["discard_count"],
            "relabel_count": counts["relabel_count"],
            "confirm_rate": round(confirm_rate, 4),
            "discard_rate": round(discard_rate, 4),
            "relabel_rate": round(relabel_rate, 4),
        }

    logger.info(f"Fallback stats for queue {queue_id}: {len(stats)} filter reasons found")
    return stats


def _adjust_ppl_multiplier(current: float, direction: str) -> float:
    adjustment = current * 0.1
    if direction == "relax":
        return round(current + adjustment, 4)
    else:
        return round(max(0.1, current - adjustment), 4)


def _adjust_similarity_threshold(current: float, direction: str) -> float:
    adjustment = current * 0.1
    if direction == "relax":
        return round(max(0.0, current - adjustment), 4)
    else:
        return round(min(1.0, current + adjustment), 4)


def _adjust_jaccard_threshold(current: float, direction: str) -> float:
    adjustment = current * 0.1
    if direction == "relax":
        return round(min(1.0, current + adjustment), 4)
    else:
        return round(max(0.0, current - adjustment), 4)


def _adjust_label_confidence_threshold(current: float, direction: str) -> float:
    adjustment = current * 0.1
    if direction == "relax":
        return round(max(0.0, current - adjustment), 4)
    else:
        return round(min(1.0, current + adjustment), 4)


PARAM_ADJUSTERS = {
    "ppl_multiplier": _adjust_ppl_multiplier,
    "similarity_threshold": _adjust_similarity_threshold,
    "jaccard_threshold": _adjust_jaccard_threshold,
    "label_confidence_threshold": _adjust_label_confidence_threshold,
}


async def _generate_recommended_filter_config(
    session: AsyncSession,
    queue: AnnotationQueue,
    annotated_version: DatasetVersion,
) -> Optional[RecommendedFilterConfig]:
    logger.info(f"[GenReco] Starting for queue={queue.id}")

    source_version_stmt = select(DatasetVersion).where(
        DatasetVersion.id == queue.version_id
    )
    source_version_result = await session.execute(source_version_stmt)
    source_version = source_version_result.scalar_one_or_none()

    if not source_version:
        logger.warning(f"[GenReco] source version {queue.version_id} not found, aborting")
        return None

    source_strictness = source_version.filter_strictness or "standard"
    base_config = FILTER_PRESETS.get(source_strictness, FILTER_PRESETS["standard"])
    logger.info(f"[GenReco] source version type={source_version.version_type}, "
                f"strictness={source_strictness}, base_config={base_config}")

    filter_stats = await _analyze_filter_reason_stats(session, queue.id)

    if not filter_stats:
        logger.info(f"[GenReco] No filter stats available, aborting")
        return None

    adjustments = {}
    reasoning_parts = []
    new_config = dict(base_config)

    for filter_reason, stats in filter_stats.items():
        param_name = FILTER_REASON_TO_PARAM.get(filter_reason)
        logger.debug(f"[GenReco] Processing {filter_reason}: stats={stats}, param_name={param_name}")

        if not param_name:
            logger.debug(f"  -> no param mapping, skip")
            continue

        if param_name not in base_config:
            logger.debug(f"  -> {param_name} not in base_config, skip")
            continue

        current_value = base_config[param_name]
        confirm_rate = stats["confirm_rate"]
        discard_rate = stats["discard_rate"]
        total = stats["total"]

        if total < 5:
            logger.debug(f"  -> total={total} < 5, skip (need >= 5 samples)")
            continue

        if confirm_rate > 0.8:
            adjuster = PARAM_ADJUSTERS.get(param_name)
            if adjuster:
                new_value = adjuster(current_value, "relax")
                new_config[param_name] = new_value
                adjustments[param_name] = {
                    "from": current_value,
                    "to": new_value,
                    "reason": f"high_confirm_rate ({confirm_rate:.1%} > 80%)",
                    "filter_reason": filter_reason,
                    "sample_count": total,
                }
                reasoning_parts.append(
                    f"{filter_reason}: confirm rate {confirm_rate:.1%} > 80%, "
                    f"{param_name} {current_value} → {new_value} (relaxed)"
                )
                logger.info(f"[GenReco]  ✓ relax {param_name}: {current_value} -> {new_value}")

        elif discard_rate > 0.6:
            adjuster = PARAM_ADJUSTERS.get(param_name)
            if adjuster:
                new_value = adjuster(current_value, "tighten")
                new_config[param_name] = new_value
                adjustments[param_name] = {
                    "from": current_value,
                    "to": new_value,
                    "reason": f"high_discard_rate ({discard_rate:.1%} > 60%)",
                    "filter_reason": filter_reason,
                    "sample_count": total,
                }
                reasoning_parts.append(
                    f"{filter_reason}: discard rate {discard_rate:.1%} > 60%, "
                    f"{param_name} {current_value} → {new_value} (tightened)"
                )
                logger.info(f"[GenReco]  ✓ tighten {param_name}: {current_value} -> {new_value}")

    if not adjustments:
        logger.info(f"[GenReco] No adjustments needed")
        return None

    logger.info(f"[GenReco] Total adjustments: {len(adjustments)}")

    try:
        recommended = RecommendedFilterConfig(
            version_id=annotated_version.id,
            queue_id=queue.id,
            source_config_name=source_strictness,
            ppl_multiplier=new_config.get("ppl_multiplier"),
            similarity_threshold=new_config.get("similarity_threshold"),
            jaccard_threshold=new_config.get("jaccard_threshold"),
            label_confidence_threshold=new_config.get("label_confidence_threshold"),
            adjustments=adjustments,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
        )
        session.add(recommended)
        await session.commit()
        await session.refresh(recommended)

        logger.info(
            f"[GenReco] SUCCESS: created RecommendedFilterConfig id={recommended.id}"
        )

        return recommended

    except Exception as e:
        logger.exception(f"[GenReco] Failed to save RecommendedFilterConfig: {e}")
        return None


async def get_recommended_filter_configs(
    session: AsyncSession,
    version_id: Optional[int] = None,
) -> list[RecommendedFilterConfig]:
    stmt = select(RecommendedFilterConfig).order_by(RecommendedFilterConfig.created_at.desc())

    if version_id is not None:
        stmt = stmt.where(RecommendedFilterConfig.version_id == version_id)

    result = await session.execute(stmt)
    return result.scalars().all()


def _compute_cohens_kappa(
    annotations_a: list[str],
    annotations_b: list[str],
) -> float:
    if len(annotations_a) != len(annotations_b) or len(annotations_a) == 0:
        return 0.0

    n = len(annotations_a)
    categories = sorted(set(annotations_a + annotations_b))

    observed_agreement = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b) / n

    freq_a = Counter(annotations_a)
    freq_b = Counter(annotations_b)
    expected_agreement = sum(
        (freq_a.get(c, 0) / n) * (freq_b.get(c, 0) / n)
        for c in categories
    )

    if expected_agreement == 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return round(kappa, 4)


def _decision_to_str(record: AnnotationRecord) -> str:
    if record.decision == AnnotationDecision.relabel:
        return f"relabel:{record.new_label or 'unknown'}"
    return record.decision.value


async def get_consistency_report(
    session: AsyncSession,
    queue_id: int,
) -> ConsistencyReport:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    records_stmt = (
        select(AnnotationRecord, AnnotationItem)
        .join(AnnotationItem, AnnotationRecord.item_id == AnnotationItem.id)
        .where(AnnotationItem.queue_id == queue_id)
    )
    records_result = await session.execute(records_stmt)
    all_rows = records_result.all()

    annotator_map: dict[str, dict[int, str]] = defaultdict(dict)
    annotator_stats: dict[str, AnnotatorStats] = defaultdict(lambda: AnnotatorStats(annotator_id=""))

    for record, item in all_rows:
        aid = record.annotator_id
        annotator_map[aid][item.id] = _decision_to_str(record)

        if aid not in annotator_stats:
            annotator_stats[aid] = AnnotatorStats(annotator_id=aid)
        annotator_stats[aid].total_annotated += 1

        if record.decision == AnnotationDecision.confirm:
            annotator_stats[aid].confirm_count += 1
        elif record.decision == AnnotationDecision.relabel:
            annotator_stats[aid].relabel_count += 1
        elif record.decision == AnnotationDecision.discard:
            annotator_stats[aid].discard_count += 1

    annotator_ids = sorted(annotator_map.keys())
    pairwise_kappa = {}
    all_kappas = []

    for i in range(len(annotator_ids)):
        for j in range(i + 1, len(annotator_ids)):
            a_id = annotator_ids[i]
            b_id = annotator_ids[j]

            common_items = sorted(set(annotator_map[a_id].keys()) & set(annotator_map[b_id].keys()))

            if common_items:
                ann_a = [annotator_map[a_id][it] for it in common_items]
                ann_b = [annotator_map[b_id][it] for it in common_items]
                kappa = _compute_cohens_kappa(ann_a, ann_b)
                pairwise_kappa[f"{a_id}_vs_{b_id}"] = kappa
                all_kappas.append(kappa)

    overall_kappa = round(sum(all_kappas) / len(all_kappas), 4) if all_kappas else 0.0

    if overall_kappa >= 0.8:
        kappa_level = "excellent"
    elif overall_kappa >= 0.6:
        kappa_level = "good"
    else:
        kappa_level = "needs_attention"

    report = ConsistencyReport(
        queue_id=queue_id,
        cohens_kappa=overall_kappa,
        kappa_level=kappa_level,
        annotator_stats=list(annotator_stats.values()),
        pairwise_kappa=pairwise_kappa,
    )

    if overall_kappa < 0.6 and len(all_kappas) > 0:
        report.warning = f"Cohen's Kappa ({overall_kappa}) is below 0.6 threshold. Annotation consistency needs attention."

    return report


async def get_annotator_performance(
    session: AsyncSession,
    annotator_id: str,
    queue_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> AnnotatorPerformanceResponse:
    stmt = select(AnnotationRecord).where(AnnotationRecord.annotator_id == annotator_id)

    if queue_id is not None:
        stmt = stmt.join(AnnotationItem, AnnotationRecord.item_id == AnnotationItem.id)
        stmt = stmt.where(AnnotationItem.queue_id == queue_id)

    if start_time or end_time:
        time_field = func.coalesce(AnnotationRecord.submitted_at, AnnotationRecord.created_at)
        if start_time:
            stmt = stmt.where(time_field >= start_time)
        if end_time:
            stmt = stmt.where(time_field <= end_time)

    result = await session.execute(stmt)
    records = result.scalars().all()

    response = AnnotatorPerformanceResponse(annotator_id=annotator_id)
    response.total_annotated = len(records)

    durations = []
    decision_counts = Counter()

    for record in records:
        if record.decision == AnnotationDecision.confirm:
            response.confirm_count += 1
        elif record.decision == AnnotationDecision.relabel:
            response.relabel_count += 1
        elif record.decision == AnnotationDecision.discard:
            response.discard_count += 1

        decision_counts[record.decision.value] += 1

        if record.annotation_duration_seconds is not None:
            durations.append(record.annotation_duration_seconds)

        if record.is_final_decision:
            response.total_agreed += 1

    if durations:
        response.avg_annotation_seconds = round(sum(durations) / len(durations), 2)
        sorted_durations = sorted(durations)
        mid = len(sorted_durations) // 2
        if len(sorted_durations) % 2 == 0:
            response.median_annotation_seconds = round(
                (sorted_durations[mid - 1] + sorted_durations[mid]) / 2, 2
            )
        else:
            response.median_annotation_seconds = round(sorted_durations[mid], 2)

    if response.total_annotated > 0:
        response.agreement_rate = round(response.total_agreed / response.total_annotated, 4)
        response.decision_distribution = dict(decision_counts)

    return response


async def get_queue_annotator_performances(
    session: AsyncSession,
    queue_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[AnnotatorPerformanceResponse]:
    stmt = (
        select(AnnotationRecord.annotator_id)
        .join(AnnotationItem, AnnotationRecord.item_id == AnnotationItem.id)
        .where(AnnotationItem.queue_id == queue_id)
    )

    if start_time or end_time:
        time_field = func.coalesce(AnnotationRecord.submitted_at, AnnotationRecord.created_at)
        if start_time:
            stmt = stmt.where(time_field >= start_time)
        if end_time:
            stmt = stmt.where(time_field <= end_time)

    stmt = stmt.distinct()
    result = await session.execute(stmt)
    annotator_ids = [row[0] for row in result.all()]

    performances = []
    for aid in annotator_ids:
        perf = await get_annotator_performance(
            session, aid, queue_id=queue_id,
            start_time=start_time, end_time=end_time
        )
        if perf.total_annotated > 0:
            performances.append(perf)

    performances.sort(key=lambda x: x.total_annotated, reverse=True)
    return performances


async def _send_webhook_notification(
    session: AsyncSession,
    queue: AnnotationQueue,
    threshold: float,
    progress: QueueProgressStats,
) -> None:
    if not queue.webhook_url:
        return

    try:
        finalized = progress.annotated + progress.arbitrated
        percentage = round(finalized / progress.total * 100, 2) if progress.total > 0 else 0.0

        decision_distribution = {}
        if finalized > 0:
            decision_distribution = {
                "confirm": round(progress.confirm_count / finalized, 4),
                "relabel": round(progress.relabel_count / finalized, 4),
                "discard": round(progress.discard_count / finalized, 4),
            }

        payload = {
            "queue_id": queue.id,
            "queue_name": queue.name,
            "threshold": threshold,
            "current_progress_percent": percentage,
            "total_samples": progress.total,
            "annotated_count": finalized,
            "pending_count": progress.pending,
            "locked_count": progress.locked,
            "decision_distribution": decision_distribution,
            "confirm_count": progress.confirm_count,
            "relabel_count": progress.relabel_count,
            "discard_count": progress.discard_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

        success = False
        status_code = None
        response_body = None
        error_message = None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(queue.webhook_url, json=payload)
                status_code = resp.status_code
                success = 200 <= resp.status_code < 300
                try:
                    response_body = resp.text[:500]
                except Exception:
                    response_body = None
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Webhook notification failed for queue {queue.id} at threshold {threshold}: {e}")

        webhook_log = WebhookLog(
            queue_id=queue.id,
            threshold=threshold,
            url=queue.webhook_url,
            status_code=status_code,
            success=success,
            response_body=response_body,
            error_message=error_message,
        )
        session.add(webhook_log)

        if success:
            triggered = list(queue.triggered_thresholds or [])
            if threshold not in triggered:
                triggered.append(threshold)
                queue.triggered_thresholds = triggered

        await session.commit()

    except Exception as e:
        logger.exception(f"Error sending webhook notification for queue {queue.id}")


async def _check_and_trigger_webhooks(
    session: AsyncSession,
    queue: AnnotationQueue,
) -> None:
    if not queue.webhook_url or not queue.webhook_thresholds:
        return

    progress = await get_queue_progress(session, queue.id)
    finalized = progress.annotated + progress.arbitrated
    percentage = (finalized / progress.total * 100) if progress.total > 0 else 0.0

    triggered = set(queue.triggered_thresholds or [])

    for threshold in queue.webhook_thresholds:
        if threshold in triggered:
            continue
        if percentage >= threshold:
            await _send_webhook_notification(session, queue, threshold, progress)


async def bulk_import_annotations(
    session: AsyncSession,
    queue_id: int,
    csv_content: str,
) -> BulkImportResult:
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()

    if not queue:
        raise ValueError("Queue not found")

    if queue.status in [QueueStatus.applied, QueueStatus.closed]:
        raise ValueError(f"Queue is {queue.status.value}, cannot import annotations")

    result_data = BulkImportResult()

    try:
        reader = csv.DictReader(io.StringIO(csv_content))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    required_columns = {"sample_id", "decision", "new_label", "annotator_id"}
    if not required_columns.issubset(set(reader.fieldnames or [])):
        missing = required_columns - set(reader.fieldnames or [])
        raise ValueError(f"CSV missing required columns: {missing}")

    item_sample_map: dict[int, AnnotationItem] = {}
    items_stmt = select(AnnotationItem).where(AnnotationItem.queue_id == queue_id)
    items_result = await session.execute(items_stmt)
    for item in items_result.scalars().all():
        item_sample_map[item.sample_id] = item

    valid_decisions = {d.value for d in AnnotationDecision}

    rows = list(reader)
    result_data.total_records = len(rows)

    import_items: list[tuple[int, str, AnnotationDecision, Optional[str]]] = []

    for i, row in enumerate(rows):
        row_num = i + 2
        try:
            sample_id = int(row["sample_id"].strip())
        except (ValueError, TypeError):
            result_data.errors.append(
                {"row": row_num, "error": f"Invalid sample_id: {row['sample_id']}"}
            )
            continue

        decision_str = row["decision"].strip().lower()
        annotator_id = row["annotator_id"].strip()
        new_label = row["new_label"].strip() if row["new_label"] else None

        if sample_id not in item_sample_map:
            result_data.sample_ids_not_found.append(sample_id)
            result_data.errors.append(
                {"row": row_num, "sample_id": sample_id, "error": "Sample not found in queue"}
            )
            continue

        if decision_str not in valid_decisions:
            result_data.invalid_decisions.append(decision_str)
            result_data.errors.append(
                {"row": row_num, "sample_id": sample_id, "error": f"Invalid decision: {decision_str}"}
            )
            continue

        decision = AnnotationDecision(decision_str)

        if decision == AnnotationDecision.relabel and not new_label:
            result_data.missing_new_labels.append(sample_id)
            result_data.errors.append(
                {"row": row_num, "sample_id": sample_id, "error": "new_label is required for relabel decision"}
            )
            continue

        if not annotator_id:
            result_data.errors.append(
                {"row": row_num, "sample_id": sample_id, "error": "annotator_id is required"}
            )
            continue

        import_items.append((sample_id, annotator_id, decision, new_label))

    for sample_id, annotator_id, decision, new_label in import_items:
        try:
            item = item_sample_map[sample_id]

            existing_stmt = select(AnnotationRecord).where(
                and_(
                    AnnotationRecord.item_id == item.id,
                    AnnotationRecord.annotator_id == annotator_id,
                )
            )
            existing_result = await session.execute(existing_stmt)
            existing_record = existing_result.scalar_one_or_none()

            now = datetime.utcnow()

            if existing_record:
                existing_record.decision = decision
                existing_record.new_label = new_label
                existing_record.submitted_at = now
            else:
                record = AnnotationRecord(
                    item_id=item.id,
                    annotator_id=annotator_id,
                    decision=decision,
                    new_label=new_label,
                    submitted_at=now,
                    created_at=now,
                )
                session.add(record)

            await session.flush()
            await _resolve_item_status(session, item, queue)

            result_data.imported_count += 1
        except Exception as e:
            logger.exception(f"Error importing annotation for sample {sample_id}")
            result_data.errors.append(
                {"sample_id": sample_id, "annotator_id": annotator_id, "error": str(e)}
            )

    await session.commit()

    await _check_and_trigger_webhooks(session, queue)

    return result_data
