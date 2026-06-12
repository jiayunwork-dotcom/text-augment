from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models.db_models import TrainingExperiment, EvaluationResult, TaskStatus
from ..models.schemas import (
    TrainingExperimentCreate, TrainingExperimentResponse,
    EvaluationResultResponse,
)
from ..services import training, evaluation

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/experiments", response_model=TrainingExperimentResponse)
async def create_training_experiment(
    request: TrainingExperimentCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    experiment = TrainingExperiment(
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        training_mode=request.training_mode,
        backbone=request.backbone,
        hyperparams=request.hyperparams.model_dump(),
        augmentation_multiplier=request.augmentation_multiplier,
        status=TaskStatus.pending,
        total_epochs=request.hyperparams.epochs,
    )
    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)

    background_tasks.add_task(_run_training, experiment.id)

    return TrainingExperimentResponse.model_validate(experiment)


async def _run_training(experiment_id: int):
    from ..database import async_session
    async with async_session() as session:
        await training.execute_training(session, experiment_id)


@router.get("/experiments", response_model=list[TrainingExperimentResponse])
async def list_experiments(
    dataset_id: int = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(TrainingExperiment).order_by(TrainingExperiment.created_at.desc())
    if dataset_id:
        stmt = stmt.where(TrainingExperiment.dataset_id == dataset_id)
    result = await session.execute(stmt)
    experiments = result.scalars().all()
    return [TrainingExperimentResponse.model_validate(e) for e in experiments]


@router.get("/experiments/{experiment_id}", response_model=TrainingExperimentResponse)
async def get_experiment(
    experiment_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(TrainingExperiment).where(TrainingExperiment.id == experiment_id)
    result = await session.execute(stmt)
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return TrainingExperimentResponse.model_validate(exp)


@router.get("/experiments/{experiment_id}/evaluation", response_model=EvaluationResultResponse)
async def get_experiment_evaluation(
    experiment_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await evaluation.get_evaluation(session, experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    return result


@router.post("/experiments/{experiment_id}/cancel")
async def cancel_experiment(
    experiment_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(TrainingExperiment).where(TrainingExperiment.id == experiment_id)
    result = await session.execute(stmt)
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status not in (TaskStatus.pending, TaskStatus.running):
        raise HTTPException(status_code=400, detail="Experiment is not running")
    exp.status = TaskStatus.failed
    exp.error_message = "Cancelled by user"
    await session.commit()
    return {"experiment_id": experiment_id, "status": "cancelled"}
