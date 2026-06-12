from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models.db_models import FilterTask, TaskStatus
from ..models.schemas import FilterTaskCreate, FilterTaskResponse
from ..services import filtering
from ..services import business_rules

router = APIRouter(prefix="/filtering", tags=["filtering"])


@router.post("/tasks", response_model=FilterTaskResponse)
async def create_filter_task(
    request: FilterTaskCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    task = FilterTask(
        version_id=request.version_id,
        strictness=request.strictness,
        status=TaskStatus.pending,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    background_tasks.add_task(_run_filtering, task.id)

    return FilterTaskResponse.model_validate(task)


async def _run_filtering(task_id: int):
    from ..database import async_session
    async with async_session() as session:
        await filtering.execute_filter_task(session, task_id)
        stmt = select(FilterTask).where(FilterTask.id == task_id)
        result = await session.execute(stmt)
        task = result.scalar_one_or_none()
        if task and task.status == TaskStatus.completed:
            await filtering.create_filtered_version(session, task)


@router.get("/tasks", response_model=list[FilterTaskResponse])
async def list_filter_tasks(
    version_id: int = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(FilterTask).order_by(FilterTask.created_at.desc())
    if version_id:
        stmt = stmt.where(FilterTask.version_id == version_id)
    result = await session.execute(stmt)
    tasks = result.scalars().all()
    return [FilterTaskResponse.model_validate(t) for t in tasks]


@router.get("/tasks/{task_id}", response_model=FilterTaskResponse)
async def get_filter_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(FilterTask).where(FilterTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Filter task not found")
    return FilterTaskResponse.model_validate(task)


@router.get("/presets")
async def get_filter_presets():
    from ..config import FILTER_PRESETS
    return FILTER_PRESETS
