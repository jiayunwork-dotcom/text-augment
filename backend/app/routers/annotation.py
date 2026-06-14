from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional

from ..database import get_session
from ..models.db_models import (
    AnnotationQueue, QueueStatus, DatasetVersion,
)
from ..models.schemas import (
    AnnotationQueueCreate, AnnotationQueueResponse,
    QueueProgressStats, AnnotationSampleResponse,
    ClaimTasksRequest, SubmitAnnotationsRequest,
    ReleaseLocksRequest, ArbitrateRequest,
    ApplyQueueRequest, ConsistencyReport,
    AnnotatorPerformanceResponse, BulkImportResult,
    RecommendedFilterConfigResponse,
)
from ..services import annotation as annotation_service

router = APIRouter(prefix="/annotation", tags=["annotation"])


@router.post("/queues", response_model=AnnotationQueueResponse)
async def create_queue(
    request: AnnotationQueueCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        queue = await annotation_service.create_annotation_queue(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    progress = await annotation_service.get_queue_progress(session, queue.id)
    response = AnnotationQueueResponse.model_validate(queue)
    response.progress = progress
    return response


@router.get("/queues", response_model=list[AnnotationQueueResponse])
async def list_queues(
    version_id: int = None,
    status: str = None,
    include_progress: bool = True,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).order_by(AnnotationQueue.created_at.desc())
    if version_id:
        stmt = stmt.where(AnnotationQueue.version_id == version_id)
    if status:
        try:
            queue_status = QueueStatus(status)
            stmt = stmt.where(AnnotationQueue.status == queue_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid queue status: {status}")

    result = await session.execute(stmt)
    queues = result.scalars().all()

    responses = []
    for q in queues:
        r = AnnotationQueueResponse.model_validate(q)
        if include_progress:
            r.progress = await annotation_service.get_queue_progress(session, q.id)
        responses.append(r)
    return responses


@router.get("/queues/{queue_id}", response_model=AnnotationQueueResponse)
async def get_queue(
    queue_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    response = AnnotationQueueResponse.model_validate(queue)
    response.progress = await annotation_service.get_queue_progress(session, queue.id)
    return response


@router.post("/queues/{queue_id}/close", response_model=AnnotationQueueResponse)
async def close_queue(
    queue_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    if queue.status == QueueStatus.applied:
        raise HTTPException(status_code=400, detail="Cannot close an applied queue")

    queue.status = QueueStatus.closed
    await session.commit()
    await session.refresh(queue)

    response = AnnotationQueueResponse.model_validate(queue)
    response.progress = await annotation_service.get_queue_progress(session, queue.id)
    return response


@router.get("/queues/{queue_id}/progress", response_model=QueueProgressStats)
async def get_queue_progress(
    queue_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    return await annotation_service.get_queue_progress(session, queue_id)


@router.post("/claim", response_model=list[AnnotationSampleResponse])
async def claim_tasks(
    request: ClaimTasksRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        tasks = await annotation_service.claim_tasks(
            session, request.queue_id, request.annotator_id, request.batch_size
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [AnnotationSampleResponse(**t) for t in tasks]


@router.post("/submit")
async def submit_annotations(
    request: SubmitAnnotationsRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await annotation_service.submit_annotations(
            session, request.queue_id, request.annotator_id, request.items
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/release")
async def release_locks(
    request: ReleaseLocksRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await annotation_service.release_locks(
            session, request.queue_id, request.annotator_id, request.item_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/queues/{queue_id}/disputed")
async def get_disputed_items(
    queue_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    return await annotation_service.get_disputed_items(session, queue_id)


@router.post("/arbitrate")
async def arbitrate_items(
    request: ArbitrateRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await annotation_service.arbitrate_items(
            session, request.queue_id, request.arbitrator_id, request.items
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/apply")
async def apply_queue(
    request: ApplyQueueRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        new_version = await annotation_service.apply_queue_results(
            session, request.queue_id, request.applied_by
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "success": True,
        "queue_id": request.queue_id,
        "new_version_id": new_version.id,
        "new_version_name": new_version.version_name,
        "total_samples": new_version.total_samples,
        "class_distribution": new_version.class_distribution,
    }


@router.get("/queues/{queue_id}/consistency", response_model=ConsistencyReport)
async def get_consistency_report(
    queue_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    try:
        report = await annotation_service.get_consistency_report(session, queue_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return report


@router.get("/queues/{queue_id}/annotators", response_model=list[AnnotatorPerformanceResponse])
async def get_queue_annotator_performances(
    queue_id: int,
    start_time: Optional[datetime] = Query(None, description="Start time for filtering (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering (ISO format)"),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
    result = await session.execute(stmt)
    queue = result.scalar_one_or_none()
    if not queue:
        raise HTTPException(status_code=404, detail="Annotation queue not found")

    performances = await annotation_service.get_queue_annotator_performances(
        session, queue_id, start_time=start_time, end_time=end_time
    )
    return performances


@router.get("/annotators/{annotator_id}/performance", response_model=AnnotatorPerformanceResponse)
async def get_annotator_performance(
    annotator_id: str,
    queue_id: Optional[int] = Query(None, description="Filter by queue ID"),
    start_time: Optional[datetime] = Query(None, description="Start time for filtering (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering (ISO format)"),
    session: AsyncSession = Depends(get_session),
):
    performance = await annotation_service.get_annotator_performance(
        session, annotator_id, queue_id=queue_id,
        start_time=start_time, end_time=end_time
    )
    if performance.total_annotated == 0 and queue_id is not None:
        stmt = select(AnnotationQueue).where(AnnotationQueue.id == queue_id)
        result = await session.execute(stmt)
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Queue not found")
    return performance


@router.post("/queues/{queue_id}/import", response_model=BulkImportResult)
async def bulk_import_annotations(
    queue_id: int,
    file: UploadFile = File(..., description="CSV file with columns: sample_id, decision, new_label, annotator_id"),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await file.read()
        csv_content = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        result = await annotation_service.bulk_import_annotations(
            session, queue_id, csv_content
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.get("/recommended-filter-configs", response_model=list[RecommendedFilterConfigResponse])
async def get_recommended_filter_configs(
    version_id: Optional[int] = Query(None, description="Filter by annotated version ID"),
    session: AsyncSession = Depends(get_session),
):
    configs = await annotation_service.get_recommended_filter_configs(
        session, version_id=version_id
    )
    return configs
