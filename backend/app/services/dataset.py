import csv
import json
import random
import io
from typing import Optional
from collections import Counter
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.db_models import Dataset, DatasetVersion, Sample, SplitType, SampleSource
from ..models.schemas import DatasetImportResponse, DatasetVersionResponse, VersionComparison
from ..config import DEFAULT_SPLIT_RATIOS, MAX_AUGMENTATION_MULTIPLIER


async def import_dataset(
    session: AsyncSession,
    name: str,
    description: str,
    file_content: bytes,
    file_format: str,
    text_column: str = "text",
    label_column: str = "label",
    split_ratios: tuple = DEFAULT_SPLIT_RATIOS,
    random_seed: int = 42,
) -> DatasetImportResponse:
    samples_data = _parse_file(file_content, file_format, text_column, label_column)
    if not samples_data:
        raise ValueError("No valid samples found in uploaded file")

    class_distribution = Counter(s["label"] for s in samples_data)
    num_classes = len(class_distribution)
    total_samples = len(samples_data)
    min_class_samples = min(class_distribution.values())
    max_class_samples = max(class_distribution.values())
    imbalance_ratio = max_class_samples / max(min_class_samples, 1)

    dataset = Dataset(
        name=name,
        description=description,
        num_classes=num_classes,
        total_samples=total_samples,
        min_class_samples=min_class_samples,
        imbalance_ratio=round(imbalance_ratio, 2),
        class_distribution=dict(class_distribution),
    )
    session.add(dataset)
    await session.flush()

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_name="original",
        version_type="original",
        total_samples=total_samples,
        class_distribution=dict(class_distribution),
        split_ratios={"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]},
    )
    session.add(version)
    await session.flush()

    splits = _assign_splits(len(samples_data), split_ratios, random_seed)
    for idx, sample_data in enumerate(samples_data):
        sample = Sample(
            version_id=version.id,
            text=sample_data["text"],
            label=sample_data["label"],
            split=splits[idx],
            source=SampleSource.original,
        )
        session.add(sample)

    await session.commit()

    return DatasetImportResponse(
        dataset_id=dataset.id,
        version_id=version.id,
        total_samples=total_samples,
        num_classes=num_classes,
        class_distribution=dict(class_distribution),
        imbalance_ratio=round(imbalance_ratio, 2),
    )


def _parse_file(file_content: bytes, file_format: str, text_column: str, label_column: str) -> list[dict]:
    samples = []
    if file_format == "csv":
        text = file_content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            text_val = row.get(text_column, "").strip()
            label_val = row.get(label_column, "").strip()
            if text_val and label_val:
                samples.append({"text": text_val, "label": label_val})
    elif file_format == "json":
        data = json.loads(file_content.decode("utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text_val = str(item.get(text_column, "")).strip()
                    label_val = str(item.get(label_column, "")).strip()
                    if text_val and label_val:
                        samples.append({"text": text_val, "label": label_val})
    return samples


def _assign_splits(n: int, ratios: tuple, seed: int) -> list[SplitType]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    splits = [SplitType.train] * n
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    for i in range(train_end, val_end):
        splits[indices[i]] = SplitType.val
    for i in range(val_end, n):
        splits[indices[i]] = SplitType.test
    return splits


async def import_unlabeled(
    session: AsyncSession,
    dataset_id: int,
    file_content: bytes,
    file_format: str,
    text_column: str = "text",
) -> dict:
    texts = _parse_unlabeled_file(file_content, file_format, text_column)
    if not texts:
        raise ValueError("No valid unlabeled samples found in uploaded file")

    stmt = select(Dataset).where(Dataset.id == dataset_id)
    result = await session.execute(stmt)
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError("Dataset not found")

    version = DatasetVersion(
        dataset_id=dataset_id,
        version_name="unlabeled",
        version_type="unlabeled",
        total_samples=len(texts),
        class_distribution={},
        split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
    )
    session.add(version)
    await session.flush()

    for text in texts:
        sample = Sample(
            version_id=version.id,
            text=text,
            label="__unlabeled__",
            split=SplitType.train,
            source=SampleSource.unlabeled,
        )
        session.add(sample)

    await session.commit()

    return {
        "dataset_id": dataset_id,
        "version_id": version.id,
        "total_samples": len(texts),
    }


def _parse_unlabeled_file(file_content: bytes, file_format: str, text_column: str) -> list[str]:
    texts = []
    if file_format == "csv":
        text = file_content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            text_val = row.get(text_column, "").strip()
            if text_val:
                texts.append(text_val)
    elif file_format == "json":
        data = json.loads(file_content.decode("utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text_val = str(item.get(text_column, "")).strip()
                    if text_val:
                        texts.append(text_val)
        elif isinstance(data, dict):
            text_val = str(data.get(text_column, "")).strip()
            if text_val:
                texts.append(text_val)
    elif file_format == "txt":
        text = file_content.decode("utf-8")
        for line in text.split("\n"):
            line = line.strip()
            if line:
                texts.append(line)
    return texts


async def list_datasets(session: AsyncSession) -> list[dict]:
    stmt = select(Dataset).options(selectinload(Dataset.versions)).order_by(Dataset.created_at.desc())
    result = await session.execute(stmt)
    datasets = result.scalars().all()
    response = []
    for ds in datasets:
        response.append({
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "num_classes": ds.num_classes,
            "total_samples": ds.total_samples,
            "min_class_samples": ds.min_class_samples,
            "imbalance_ratio": ds.imbalance_ratio,
            "class_distribution": ds.class_distribution,
            "version_count": len(ds.versions),
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
        })
    return response


async def get_dataset(session: AsyncSession, dataset_id: int) -> Optional[dict]:
    stmt = select(Dataset).where(Dataset.id == dataset_id).options(selectinload(Dataset.versions))
    result = await session.execute(stmt)
    ds = result.scalar_one_or_none()
    if not ds:
        return None
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "num_classes": ds.num_classes,
        "total_samples": ds.total_samples,
        "min_class_samples": ds.min_class_samples,
        "imbalance_ratio": ds.imbalance_ratio,
        "class_distribution": ds.class_distribution,
        "version_count": len(ds.versions),
        "versions": [
            {
                "id": v.id,
                "version_name": v.version_name,
                "version_type": v.version_type,
                "total_samples": v.total_samples,
                "class_distribution": v.class_distribution,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in ds.versions
        ],
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


async def list_versions(session: AsyncSession, dataset_id: int) -> list[dict]:
    stmt = select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id).order_by(DatasetVersion.created_at)
    result = await session.execute(stmt)
    versions = result.scalars().all()
    return [
        {
            "id": v.id,
            "dataset_id": v.dataset_id,
            "version_name": v.version_name,
            "version_type": v.version_type,
            "total_samples": v.total_samples,
            "class_distribution": v.class_distribution,
            "split_ratios": v.split_ratios,
            "filter_strictness": v.filter_strictness,
            "parent_version_id": v.parent_version_id,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


async def get_version_samples(
    session: AsyncSession,
    version_id: int,
    split: Optional[str] = None,
    source: Optional[str] = None,
    is_filtered: Optional[bool] = None,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    stmt = select(Sample).where(Sample.version_id == version_id)
    count_stmt = select(func.count()).select_from(Sample).where(Sample.version_id == version_id)

    if split:
        stmt = stmt.where(Sample.split == split)
        count_stmt = count_stmt.where(Sample.split == split)
    if source:
        stmt = stmt.where(Sample.source == source)
        count_stmt = count_stmt.where(Sample.source == source)
    if is_filtered is not None:
        stmt = stmt.where(Sample.is_filtered == is_filtered)
        count_stmt = count_stmt.where(Sample.is_filtered == is_filtered)

    total_result = await session.execute(count_stmt)
    total = total_result.scalar()

    stmt = stmt.offset(offset).limit(limit)
    result = await session.execute(stmt)
    samples = result.scalars().all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "samples": [
            {
                "id": s.id,
                "text": s.text,
                "label": s.label,
                "split": s.split.value if s.split else None,
                "source": s.source.value if s.source else None,
                "is_filtered": s.is_filtered,
                "filter_reason": s.filter_reason,
                "is_manually_approved": s.is_manually_approved,
                "perplexity": s.perplexity,
                "similarity_score": s.similarity_score,
                "label_confidence": s.label_confidence,
            }
            for s in samples
        ],
    }


async def compare_versions(session: AsyncSession, version_id_a: int, version_id_b: int) -> VersionComparison:
    stmt = select(DatasetVersion).where(DatasetVersion.id.in_([version_id_a, version_id_b]))
    result = await session.execute(stmt)
    versions = {v.id: v for v in result.scalars().all()}

    if version_id_a not in versions or version_id_b not in versions:
        raise ValueError("One or both versions not found")

    va = versions[version_id_a]
    vb = versions[version_id_b]

    all_keys = set(list(va.class_distribution.keys()) + list(vb.class_distribution.keys()))
    distribution_diff = {}
    for key in all_keys:
        count_a = va.class_distribution.get(key, 0)
        count_b = vb.class_distribution.get(key, 0)
        distribution_diff[key] = {"version_a": count_a, "version_b": count_b, "diff": count_b - count_a}

    return VersionComparison(
        version_a_id=version_id_a,
        version_b_id=version_id_b,
        sample_count_diff=vb.total_samples - va.total_samples,
        distribution_diff=distribution_diff,
    )


async def approve_samples(session: AsyncSession, sample_ids: list[int]) -> int:
    stmt = select(Sample).where(Sample.id.in_(sample_ids))
    result = await session.execute(stmt)
    samples = result.scalars().all()
    count = 0
    for s in samples:
        s.is_manually_approved = True
        s.is_filtered = False
        s.filter_reason = None
        count += 1
    await session.commit()
    return count


async def resplit_version(session: AsyncSession, version_id: int, train_ratio: float, val_ratio: float, test_ratio: float, random_seed: int = 42) -> dict:
    stmt = select(Sample).where(Sample.version_id == version_id)
    result = await session.execute(stmt)
    samples = result.scalars().all()

    rng = random.Random(random_seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)

    for i, idx in enumerate(indices):
        if i < train_end:
            samples[idx].split = SplitType.train
        elif i < val_end:
            samples[idx].split = SplitType.val
        else:
            samples[idx].split = SplitType.test

    split_counts = Counter(s.split.value for s in samples)
    await session.commit()

    return {"version_id": version_id, "split_counts": dict(split_counts)}


async def create_version(session: AsyncSession, dataset_id: int, version_name: str, version_type: str, parent_version_id: Optional[int] = None) -> DatasetVersion:
    version = DatasetVersion(
        dataset_id=dataset_id,
        version_name=version_name,
        version_type=version_type,
        parent_version_id=parent_version_id,
    )
    session.add(version)
    await session.commit()
    await session.refresh(version)
    return version


async def delete_dataset(session: AsyncSession, dataset_id: int) -> bool:
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    result = await session.execute(stmt)
    ds = result.scalar_one_or_none()
    if not ds:
        return False
    await session.delete(ds)
    await session.commit()
    return True
