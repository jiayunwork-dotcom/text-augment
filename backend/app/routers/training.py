from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models.db_models import TrainingExperiment, EvaluationResult, TaskStatus, TrainingMode
from ..models.schemas import (
    TrainingExperimentCreate, TrainingExperimentResponse,
    EvaluationResultResponse,
)
from ..services import training, evaluation
from ..services import business_rules

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/experiments", response_model=TrainingExperimentResponse)
async def create_training_experiment(
    request: TrainingExperimentCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    from ..models.db_models import DatasetVersion

    version_stmt = select(DatasetVersion).where(DatasetVersion.id == request.version_id)
    version_result = await session.execute(version_stmt)
    version = version_result.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    filter_check = await business_rules.check_version_filtered(session, request.version_id)
    if not filter_check["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": filter_check["reason"],
                "required_action": filter_check.get("required_action", ""),
                "version_type": filter_check.get("version_type", ""),
            },
        )

    if request.training_mode == TrainingMode.semi_supervised:
        if not request.unlabeled_version_id:
            raise HTTPException(
                status_code=400,
                detail="Semi-supervised training requires `unlabeled_version_id` to be specified. "
                       "Please import unlabeled data first.",
            )
        unlabeled_stmt = select(DatasetVersion).where(DatasetVersion.id == request.unlabeled_version_id)
        unlabeled_result = await session.execute(unlabeled_stmt)
        unlabeled_version = unlabeled_result.scalar_one_or_none()
        if not unlabeled_version:
            raise HTTPException(status_code=404, detail="Unlabeled dataset version not found")
        if unlabeled_version.version_type != "unlabeled":
            raise HTTPException(
                status_code=400,
                detail=f"unlabeled_version_id refers to version of type '{unlabeled_version.version_type}'. "
                       "It must be of type 'unlabeled'.",
            )

    experiment = TrainingExperiment(
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        unlabeled_version_id=request.unlabeled_version_id,
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
