import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Enum, ForeignKey,
    JSON, Boolean, Index
)
from sqlalchemy.orm import relationship
from ..database import Base


class SplitType(str, enum.Enum):
    train = "train"
    val = "val"
    test = "test"


class TaskStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


class FilterStrictness(str, enum.Enum):
    loose = "loose"
    standard = "standard"
    strict = "strict"


class TrainingMode(str, enum.Enum):
    baseline = "baseline"
    augmented = "augmented"
    curriculum = "curriculum"
    semi_supervised = "semi_supervised"


class ModelBackbone(str, enum.Enum):
    distilbert = "distilbert"
    tinybert = "tinybert"
    textcnn = "textcnn"
    bilstm_attention = "bilstm_attention"


class SampleSource(str, enum.Enum):
    original = "original"
    synonym_replacement = "synonym_replacement"
    random_ops = "random_ops"
    back_translation = "back_translation"
    context_augment = "context_augment"
    template_generation = "template_generation"
    oversampling = "oversampling"
    pseudo_label = "pseudo_label"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    num_classes = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    min_class_samples = Column(Integer, default=0)
    imbalance_ratio = Column(Float, default=1.0)
    class_distribution = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_name = Column(String(100), nullable=False)
    version_type = Column(String(50), default="original")
    total_samples = Column(Integer, default=0)
    class_distribution = Column(JSON, default=dict)
    split_ratios = Column(JSON, default=dict)
    filter_strictness = Column(String(20), default="standard")
    parent_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")
    samples = relationship("Sample", back_populates="version", cascade="all, delete-orphan")
    parent_version = relationship("DatasetVersion", remote_side=[id])


class Sample(Base):
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    label = Column(String(100), nullable=False)
    split = Column(Enum(SplitType), default=SplitType.train)
    source = Column(Enum(SampleSource), default=SampleSource.original)
    source_sample_id = Column(Integer, nullable=True)
    is_filtered = Column(Boolean, default=False)
    filter_reason = Column(String(100), nullable=True)
    is_manually_approved = Column(Boolean, default=False)
    perplexity = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)
    label_confidence = Column(Float, nullable=True)

    version = relationship("DatasetVersion", back_populates="samples")

    __table_args__ = (Index("idx_sample_version_split", "version_id", "split"),)


class AugmentationTask(Base):
    __tablename__ = "augmentation_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    source_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    target_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    strategy = Column(String(50), nullable=False)
    strategy_params = Column(JSON, default=dict)
    status = Column(Enum(TaskStatus), default=TaskStatus.pending)
    total_samples = Column(Integer, default=0)
    processed_samples = Column(Integer, default=0)
    generated_samples = Column(Integer, default=0)
    augmentation_multiplier = Column(Float, default=1.0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    estimated_remaining_seconds = Column(Float, nullable=True)

    source_version = relationship("DatasetVersion", foreign_keys=[source_version_id])
    target_version = relationship("DatasetVersion", foreign_keys=[target_version_id])


class FilterTask(Base):
    __tablename__ = "filter_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    target_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    strictness = Column(Enum(FilterStrictness), default=FilterStrictness.standard)
    status = Column(Enum(TaskStatus), default=TaskStatus.pending)
    total_samples = Column(Integer, default=0)
    passed_samples = Column(Integer, default=0)
    filtered_samples = Column(Integer, default=0)
    ppl_filtered = Column(Integer, default=0)
    label_filtered = Column(Integer, default=0)
    similarity_filtered = Column(Integer, default=0)
    dedup_filtered = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingExperiment(Base):
    __tablename__ = "training_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(255), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    training_mode = Column(Enum(TrainingMode), nullable=False)
    backbone = Column(Enum(ModelBackbone), nullable=False)
    hyperparams = Column(JSON, default=dict)
    augmentation_multiplier = Column(Float, default=1.0)
    status = Column(Enum(TaskStatus), default=TaskStatus.pending)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=10)
    train_loss_history = Column(JSON, default=list)
    val_loss_history = Column(JSON, default=list)
    val_metric_history = Column(JSON, default=list)
    best_epoch = Column(Integer, nullable=True)
    best_val_metric = Column(Float, nullable=True)
    model_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("training_experiments.id", ondelete="CASCADE"), nullable=False)
    accuracy = Column(Float, default=0.0)
    macro_f1 = Column(Float, default=0.0)
    weighted_f1 = Column(Float, default=0.0)
    per_class_metrics = Column(JSON, default=dict)
    test_loss = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("TrainingExperiment")


class ComparisonStudy(Base):
    __tablename__ = "comparison_studies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    comparison_type = Column(String(50), nullable=False)
    experiment_ids = Column(JSON, default=list)
    results = Column(JSON, default=dict)
    significance_tests = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
