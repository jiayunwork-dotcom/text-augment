from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models.schemas import (
    LearningCurveRequest, StrategyComparisonRequest,
    AugmentationRatioAnalysisRequest, SignificanceTestRequest,
    SignificanceTestResponse,
)
from ..services import evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/learning-curve")
async def learning_curve(
    request: LearningCurveRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        results = await evaluation.run_learning_curve(
            session=session,
            dataset_id=request.dataset_id,
            version_id=request.version_id,
            backbone=request.backbone,
            training_mode=request.training_mode,
            data_fractions=request.data_fractions,
            hyperparams=request.hyperparams.model_dump(),
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-strategies")
async def compare_strategies(
    request: StrategyComparisonRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await evaluation.compare_strategies(session, request.version_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/significance-test", response_model=SignificanceTestResponse)
async def significance_test(
    request: SignificanceTestRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await evaluation.significance_test(
            session=session,
            experiment_id_a=request.experiment_id_a,
            experiment_id_b=request.experiment_id_b,
            test_type=request.test_type,
            num_bootstrap=request.num_bootstrap,
        )
        return SignificanceTestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await evaluation.get_evaluation(session, experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    return result
