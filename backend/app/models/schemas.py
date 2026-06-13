from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class SplitType(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


class FilterStrictness(str, Enum):
    loose = "loose"
    standard = "standard"
    strict = "strict"


class TrainingMode(str, Enum):
    baseline = "baseline"
    augmented = "augmented"
    curriculum = "curriculum"
    semi_supervised = "semi_supervised"


class ModelBackbone(str, Enum):
    distilbert = "distilbert"
    tinybert = "tinybert"
    textcnn = "textcnn"
    bilstm_attention = "bilstm_attention"


class SampleSource(str, Enum):
    original = "original"
    synonym_replacement = "synonym_replacement"
    random_ops = "random_ops"
    back_translation = "back_translation"
    context_augment = "context_augment"
    template_generation = "template_generation"
    oversampling = "oversampling"
    pseudo_label = "pseudo_label"
    unlabeled = "unlabeled"


class DatasetCreate(BaseModel):
    name: str
    description: str = ""


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str
    num_classes: int
    total_samples: int
    min_class_samples: int
    imbalance_ratio: float
    class_distribution: dict
    version_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetVersionResponse(BaseModel):
    id: int
    dataset_id: int
    version_name: str
    version_type: str
    total_samples: int
    class_distribution: dict
    split_ratios: dict
    filter_strictness: str
    parent_version_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class VersionComparison(BaseModel):
    version_a_id: int
    version_b_id: int
    sample_count_diff: int
    distribution_diff: dict


class DatasetImportResponse(BaseModel):
    dataset_id: int
    version_id: int
    total_samples: int
    num_classes: int
    class_distribution: dict
    imbalance_ratio: float


class UnlabeledImportResponse(BaseModel):
    dataset_id: int
    version_id: int
    total_samples: int


class SplitRequest(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


class SampleResponse(BaseModel):
    id: int
    text: str
    label: str
    split: SplitType
    source: SampleSource
    is_filtered: bool
    filter_reason: Optional[str] = None
    is_manually_approved: bool
    perplexity: Optional[float] = None
    similarity_score: Optional[float] = None
    label_confidence: Optional[float] = None

    class Config:
        from_attributes = True


class SynonymReplaceParams(BaseModel):
    replace_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    language: str = "en"


class RandomOpsParams(BaseModel):
    n_ops: Optional[int] = None
    insert_ratio: float = 0.25
    swap_ratio: float = 0.25
    delete_ratio: float = 0.25
    delete_prob: float = 0.1
    language: str = "en"


class BackTranslationParams(BaseModel):
    pivot_language: str = "fr"
    num_variants: int = Field(default=1, ge=1, le=5)
    source_language: str = "en"


class ContextAugmentParams(BaseModel):
    mask_ratio: float = Field(default=0.15, ge=0.0, le=0.5)
    top_k: int = Field(default=5, ge=1, le=20)
    num_variants: int = Field(default=1, ge=1, le=10)
    model_name: str = "bert-base-uncased"


class TemplateGenerateParams(BaseModel):
    template: str = '{label}类的例子: {text}'
    samples_per_seed: int = Field(default=3, ge=1, le=20)
    seed_count: Optional[int] = None


class AugmentationTaskCreate(BaseModel):
    dataset_id: int
    source_version_id: int
    strategy: str
    strategy_params: dict = {}
    augmentation_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)


class AugmentationTaskResponse(BaseModel):
    id: int
    dataset_id: int
    source_version_id: int
    target_version_id: Optional[int] = None
    strategy: str
    strategy_params: dict
    status: TaskStatus
    total_samples: int
    processed_samples: int
    generated_samples: int
    augmentation_multiplier: float
    error_message: Optional[str] = None
    estimated_remaining_seconds: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FilterTaskCreate(BaseModel):
    version_id: int
    strictness: FilterStrictness = FilterStrictness.standard


class FilterTaskResponse(BaseModel):
    id: int
    version_id: int
    target_version_id: Optional[int] = None
    strictness: FilterStrictness
    status: TaskStatus
    total_samples: int
    passed_samples: int
    filtered_samples: int
    ppl_filtered: int
    label_filtered: int
    similarity_filtered: int
    dedup_filtered: int
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingHyperparams(BaseModel):
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 10
    early_stopping_patience: int = 3
    max_seq_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


class TrainingExperimentCreate(BaseModel):
    experiment_name: str
    dataset_id: int
    version_id: int
    training_mode: TrainingMode
    backbone: ModelBackbone
    hyperparams: TrainingHyperparams = TrainingHyperparams()
    augmentation_multiplier: float = 1.0
    unlabeled_version_id: Optional[int] = None


class TrainingExperimentResponse(BaseModel):
    id: int
    experiment_name: str
    dataset_id: int
    version_id: int
    training_mode: TrainingMode
    backbone: ModelBackbone
    hyperparams: dict
    augmentation_multiplier: float
    status: TaskStatus
    current_epoch: int
    total_epochs: int
    train_loss_history: list
    val_loss_history: list
    val_metric_history: list
    best_epoch: Optional[int] = None
    best_val_metric: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class EvaluationResultResponse(BaseModel):
    id: int
    experiment_id: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict
    confusion_matrix: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class LearningCurveRequest(BaseModel):
    dataset_id: int
    version_id: int
    backbone: ModelBackbone
    training_mode: TrainingMode
    data_fractions: list[float] = [0.1, 0.2, 0.5, 1.0]
    hyperparams: TrainingHyperparams = TrainingHyperparams()


class StrategyComparisonRequest(BaseModel):
    dataset_id: int
    version_ids: list[int]
    backbone: ModelBackbone
    hyperparams: TrainingHyperparams = TrainingHyperparams()


class AugmentationRatioAnalysisRequest(BaseModel):
    dataset_id: int
    source_version_id: int
    strategy: str
    strategy_params: dict = {}
    ratios: list[float] = [1.0, 2.0, 3.0, 5.0]
    backbone: ModelBackbone
    hyperparams: TrainingHyperparams = TrainingHyperparams()


class SignificanceTestRequest(BaseModel):
    experiment_id_a: int
    experiment_id_b: int
    test_type: str = "paired_t"
    num_bootstrap: int = 10000


class SignificanceTestResponse(BaseModel):
    test_type: str
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: Optional[list[float]] = None


class SampleApprovalRequest(BaseModel):
    sample_ids: list[int]


class TaskActionRequest(BaseModel):
    action: str
