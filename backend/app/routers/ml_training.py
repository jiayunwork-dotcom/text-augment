from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_session
from ..models.db_models import MLTrainingStatus, MLModelType
from ..models.schemas import (
    MLTrainingTaskCreate,
    MLTrainingTaskResponse,
    MLTrainingReportResponse,
    MLModelCompareRequest,
    MLModelCompareResponse,
    MLPredictSingleRequest,
    MLPredictSingleResponse,
    MLPredictBatchRequest,
    MLPredictBatchResponse,
    MLDataLineageResponse,
    MLTrainingTaskPatch,
    MLRetryResponse,
)
from ..services import ml_training as ml_service

router = APIRouter(prefix="/ml-training", tags=["ml-training"])


async def _run_ml_training_background(task_id: int):
    from ..database import async_session
    async with async_session() as session:
        await ml_service.execute_ml_training(session, task_id)


@router.post("/tasks", response_model=MLTrainingTaskResponse)
async def create_training_task(
    request: MLTrainingTaskCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    hp = request.hyperparams.model_dump()
    splits = request.split_ratios.model_dump()

    try:
        task = await ml_service.create_ml_training_task(
            session=session,
            task_name=request.task_name,
            annotated_version_id=request.annotated_version_id,
            model_type=request.model_type,
            hyperparams=hp,
            split_ratios=splits,
            notes=request.notes,
            tags=request.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(_run_ml_training_background, task.id)
    return MLTrainingTaskResponse.model_validate(task)


@router.get("/tasks", response_model=list[MLTrainingTaskResponse])
async def list_training_tasks(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    status: Optional[MLTrainingStatus] = Query(None, description="Filter by training status"),
    model_type: Optional[MLModelType] = Query(None, description="Filter by model type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    session: AsyncSession = Depends(get_session),
):
    tasks = await ml_service.list_ml_training_tasks(session, dataset_id=dataset_id, status=status, tag=tag)
    if model_type is not None:
        tasks = [t for t in tasks if t.model_type == model_type]
    return [MLTrainingTaskResponse.model_validate(t) for t in tasks]


@router.get("/tasks/{task_id}", response_model=MLTrainingTaskResponse)
async def get_training_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    task = await ml_service.get_ml_training_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    return MLTrainingTaskResponse.model_validate(task)


@router.post("/tasks/{task_id}/cancel", response_model=MLTrainingTaskResponse)
async def cancel_training_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    try:
        task = await ml_service.cancel_ml_training_task(session, task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MLTrainingTaskResponse.model_validate(task)


@router.post("/tasks/{task_id}/retry", response_model=MLRetryResponse)
async def retry_training_task(
    task_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    try:
        new_task = await ml_service.retry_ml_training_task(session, task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    background_tasks.add_task(_run_ml_training_background, new_task.id)
    return MLRetryResponse(
        original_task_id=task_id,
        new_task_id=new_task.id,
        message="Retry task created successfully",
    )


@router.patch("/tasks/{task_id}", response_model=MLTrainingTaskResponse)
async def patch_training_task(
    task_id: int,
    request: MLTrainingTaskPatch,
    session: AsyncSession = Depends(get_session),
):
    try:
        task = await ml_service.update_ml_training_task_metadata(
            session=session,
            task_id=task_id,
            notes=request.notes,
            tags=request.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MLTrainingTaskResponse.model_validate(task)


@router.get("/tasks/{task_id}/report", response_model=MLTrainingReportResponse)
async def get_training_report(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    task = await ml_service.get_ml_training_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != MLTrainingStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed (status: {task.status.value}). Report will be available when training completes.",
        )
    report = await ml_service.get_ml_training_report(session, task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Training report not found")
    return MLTrainingReportResponse.model_validate(report)


@router.post("/compare", response_model=MLModelCompareResponse)
async def compare_models(
    request: MLModelCompareRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ml_service.compare_ml_models(session, request.task_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MLModelCompareResponse(**result)


@router.post("/predict", response_model=MLPredictSingleResponse)
async def predict_single(
    request: MLPredictSingleRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ml_service.predict_single(session, request.task_id, request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MLPredictSingleResponse(**result)


@router.post("/predict-batch", response_model=MLPredictBatchResponse)
async def predict_batch(
    request: MLPredictBatchRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ml_service.predict_batch(session, request.task_id, request.texts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MLPredictBatchResponse(**result)


@router.get("/tasks/{task_id}/lineage", response_model=MLDataLineageResponse)
async def get_data_lineage(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ml_service.get_data_lineage(session, task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return MLDataLineageResponse(**result)
