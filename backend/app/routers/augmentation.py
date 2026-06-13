import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models.db_models import AugmentationTask, TaskStatus, AugmentationStep
from ..models.schemas import (
    AugmentationTaskCreate, AugmentationTaskResponse, TaskActionRequest,
    PreviewRequest, PreviewResponse,
)
from ..services import augmentation as aug_service
from ..services import business_rules

router = APIRouter(prefix="/augmentation", tags=["augmentation"])


@router.post("/tasks", response_model=AugmentationTaskResponse)
async def create_augmentation_task(
    request: AugmentationTaskCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    is_composite = request.is_composite and request.steps

    if is_composite:
        steps_dicts = [{"strategy": s.strategy, "strategy_params": s.strategy_params} for s in request.steps]
        validation = aug_service.validate_composite_strategy(steps_dicts)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["reason"])

        for step in request.steps:
            if step.strategy == "back_translation":
                source_lang = step.strategy_params.get("source_language", "en")
                pivot_lang = step.strategy_params.get("pivot_language", "fr")
                bt_validation = business_rules.validate_back_translation_pair(source_lang, pivot_lang)
                if not bt_validation["valid"]:
                    raise HTTPException(status_code=400, detail=bt_validation["reason"])

        task = AugmentationTask(
            dataset_id=request.dataset_id,
            source_version_id=request.source_version_id,
            strategy="composite",
            strategy_params={},
            augmentation_multiplier=request.augmentation_multiplier,
            status=TaskStatus.pending,
            is_composite=True,
        )
        session.add(task)
        await session.flush()

        for i, step in enumerate(request.steps):
            db_step = AugmentationStep(
                task_id=task.id,
                step_order=i,
                strategy=step.strategy,
                strategy_params=step.strategy_params,
            )
            session.add(db_step)

        await session.commit()
        await session.refresh(task)
    else:
        if request.strategy == "back_translation":
            source_lang = request.strategy_params.get("source_language", "en")
            pivot_lang = request.strategy_params.get("pivot_language", "fr")
            validation = business_rules.validate_back_translation_pair(source_lang, pivot_lang)
            if not validation["valid"]:
                raise HTTPException(status_code=400, detail=validation["reason"])

        task = AugmentationTask(
            dataset_id=request.dataset_id,
            source_version_id=request.source_version_id,
            strategy=request.strategy,
            strategy_params=request.strategy_params,
            augmentation_multiplier=request.augmentation_multiplier,
            status=TaskStatus.pending,
            is_composite=False,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)

    background_tasks.add_task(_run_augmentation, task.id)

    return AugmentationTaskResponse.model_validate(task)


async def _run_augmentation(task_id: int):
    from ..database import async_session
    async with async_session() as session:
        await aug_service.execute_augmentation_task(session, task_id)


@router.get("/tasks", response_model=list[AugmentationTaskResponse])
async def list_augmentation_tasks(
    dataset_id: int = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AugmentationTask).order_by(AugmentationTask.created_at.desc())
    if dataset_id:
        stmt = stmt.where(AugmentationTask.dataset_id == dataset_id)
    result = await session.execute(stmt)
    tasks = result.scalars().all()
    return [AugmentationTaskResponse.model_validate(t) for t in tasks]


@router.get("/tasks/{task_id}", response_model=AugmentationTaskResponse)
async def get_augmentation_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy.orm import selectinload
    stmt = select(AugmentationTask).where(AugmentationTask.id == task_id).options(
        selectinload(AugmentationTask.steps)
    )
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return AugmentationTaskResponse.model_validate(task)


@router.post("/tasks/{task_id}/action")
async def task_action(
    task_id: int,
    request: TaskActionRequest,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AugmentationTask).where(AugmentationTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if request.action == "pause":
        await aug_service.pause_task(task_id)
        task.status = TaskStatus.paused
    elif request.action == "resume":
        await aug_service.resume_task(task_id)
        task.status = TaskStatus.running
    elif request.action == "cancel":
        await aug_service.cancel_task(task_id)
        task.status = TaskStatus.failed
        task.error_message = "Cancelled by user"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    await session.commit()
    return {"task_id": task_id, "status": task.status.value}


@router.post("/preview", response_model=PreviewResponse)
async def preview_augmentation(
    request: PreviewRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await aug_service.preview_augmentation(
            session=session,
            source_version_id=request.source_version_id,
            strategy=request.strategy,
            strategy_params=request.strategy_params,
        )
        return PreviewResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger = __import__("logging").getLogger(__name__)
        logger.exception(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
