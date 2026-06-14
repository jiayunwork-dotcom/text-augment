import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    AnnotationQueue, AnnotationItem, AnnotationRecord,
    QueueStatus, AnnotationStatus, AnnotationDecision,
    DatasetVersion, Sample, Dataset,
)
from ..models.schemas import (
    AnnotationQueueCreate, QueueProgressStats,
    ConsistencyReport, AnnotatorStats,
)

logger = logging.getLogger(__name__)


def compute_uncertainty_score(sample: Sample) -> float:
    scores = []

    if sample.label_confidence is not None:
        if 0.5 <= sample.label_confidence <= 0.8:
            scores.append(1.0 - abs(0.65 - sample.label_confidence) / 0.3)
        elif sample.label_confidence < 0.5:
            scores.append(1.0)
        else:
            scores.append(0.0)

    if sample.is_filtered and sample.filter_reason:
        scores.append(0.8)

    if sample.similarity_score is not None:
        if sample.similarity_score < 0.7:
            scores.append(0.7)
        else:
            scores.append(max(0.0, (0.85 - sample.similarity_score) / 0.3))

    if sample.perplexity is not None:
        scores.append(min(1.0, sample.perplexity / 50.0))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


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
        if score > 0.0 or s.is_filtered or (s.label_confidence is not None and 0.5 <= s.label_confidence <= 0.8):
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
    await session.commit()
    await session.refresh(queue)

    return queue


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
            else:
                record = AnnotationRecord(
                    item_id=item.id,
                    annotator_id=annotator_id,
                    decision=submit_item.decision,
                    new_label=submit_item.new_label,
                    comment=submit_item.comment,
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

    if queue.status != QueueStatus.completed:
        progress = await get_queue_progress(session, queue_id)
        remaining = progress.pending + progress.locked + progress.disputed
        raise ValueError(
            f"Queue not fully completed. Remaining unprocessed: {remaining} "
            f"(pending: {progress.pending}, locked: {progress.locked}, disputed: {progress.disputed})"
        )

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
    return new_version


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
