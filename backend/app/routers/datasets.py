from fastapi import APIRouter, UploadFile, File, Form, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_session
from ..services import dataset as dataset_service
from ..models.schemas import (
    DatasetImportResponse, DatasetResponse, SplitRequest,
    SampleApprovalRequest, VersionComparison, UnlabeledImportResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/import", response_model=DatasetImportResponse)
async def import_dataset(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
    train_ratio: float = Form(0.7),
    val_ratio: float = Form(0.15),
    test_ratio: float = Form(0.15),
    random_seed: int = Form(42),
    session: AsyncSession = Depends(get_session),
):
    content = await file.read()
    file_format = file.filename.split(".")[-1].lower() if file.filename else "csv"
    if file_format not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="File format must be csv or json")

    try:
        result = await dataset_service.import_dataset(
            session=session,
            name=name,
            description=description,
            file_content=content,
            file_format=file_format,
            text_column=text_column,
            label_column=label_column,
            split_ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=random_seed,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/import-unlabeled", response_model=UnlabeledImportResponse)
async def import_unlabeled_data(
    dataset_id: int,
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    session: AsyncSession = Depends(get_session),
):
    content = await file.read()
    filename = file.filename or ""
    file_format = filename.split(".")[-1].lower()
    if file_format not in ("csv", "json", "txt"):
        raise HTTPException(status_code=400, detail="File format must be csv, json or txt")

    try:
        result = await dataset_service.import_unlabeled(
            session=session,
            dataset_id=dataset_id,
            file_content=content,
            file_format=file_format,
            text_column=text_column,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_datasets(session: AsyncSession = Depends(get_session)):
    return await dataset_service.list_datasets(session)


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int, session: AsyncSession = Depends(get_session)):
    result = await dataset_service.get_dataset(session, dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return result


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, session: AsyncSession = Depends(get_session)):
    success = await dataset_service.delete_dataset(session, dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset deleted"}


@router.get("/{dataset_id}/versions")
async def list_versions(dataset_id: int, session: AsyncSession = Depends(get_session)):
    return await dataset_service.list_versions(session, dataset_id)


@router.get("/versions/{version_id}/samples")
async def get_version_samples(
    version_id: int,
    split: Optional[str] = None,
    source: Optional[str] = None,
    is_filtered: Optional[bool] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
):
    return await dataset_service.get_version_samples(
        session, version_id, split, source, is_filtered, offset, limit
    )


@router.post("/versions/compare")
async def compare_versions(
    version_id_a: int,
    version_id_b: int,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await dataset_service.compare_versions(session, version_id_a, version_id_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/versions/{version_id}/resplit")
async def resplit_version(
    version_id: int,
    request: SplitRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await dataset_service.resplit_version(
            session, version_id,
            request.train_ratio, request.val_ratio, request.test_ratio,
            request.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/samples/approve")
async def approve_samples(
    request: SampleApprovalRequest,
    session: AsyncSession = Depends(get_session),
):
    count = await dataset_service.approve_samples(session, request.sample_ids)
    return {"approved_count": count}
