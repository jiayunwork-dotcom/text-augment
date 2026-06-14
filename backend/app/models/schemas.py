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


class AugmentationStepCreate(BaseModel):
    strategy: str
    strategy_params: dict = {}


class AugmentationStepResponse(BaseModel):
    id: int
    task_id: int
    step_order: int
    strategy: str
    strategy_params: dict
    input_count: int = 0
    success_count: int = 0
    skipped_count: int = 0

    class Config:
        from_attributes = True


class StepStat(BaseModel):
    step_order: int
    strategy: str
    input_count: int
    success_count: int
    skipped_count: int


class AugmentationTaskCreate(BaseModel):
    dataset_id: int
    source_version_id: int
    strategy: str
    strategy_params: dict = {}
    augmentation_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    is_composite: bool = False
    steps: list[AugmentationStepCreate] = []


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
    is_composite: bool = False
    current_step_index: int = 0
    step_stats: list[StepStat] = []
    steps: list[AugmentationStepResponse] = []

    class Config:
        from_attributes = True


class PreviewRequest(BaseModel):
    source_version_id: int
    strategy: str
    strategy_params: dict = {}


class PreviewSampleResult(BaseModel):
    original_text: str
    augmented_text: Optional[str] = None
    timed_out: bool = False
    error: Optional[str] = None


class PreviewResponse(BaseModel):
    strategy: str
    strategy_params: dict
    samples: list[PreviewSampleResult]
    total_count: int
    success_count: int
    timed_out_count: int


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


class QueueStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    applied = "applied"
    closed = "closed"


class PriorityStrategy(str, Enum):
    uncertainty = "uncertainty"
    class_balance = "class_balance"
    hybrid = "hybrid"


class AnnotationDecision(str, Enum):
    confirm = "confirm"
    relabel = "relabel"
    discard = "discard"


class AnnotationStatus(str, Enum):
    pending = "pending"
    locked = "locked"
    annotated = "annotated"
    disputed = "disputed"
    arbitrated = "arbitrated"


class AnnotationQueueCreate(BaseModel):
    version_id: int
    name: str = ""
    capacity: int = Field(default=100, ge=1, le=10000)
    review_mode: str = Field(default="single", pattern="^(single|multi)$")
    num_reviewers: int = Field(default=1, ge=1, le=9)
    lock_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    created_by: str = ""
    priority_strategy: PriorityStrategy = PriorityStrategy.uncertainty
    webhook_url: Optional[str] = None
    webhook_thresholds: list[float] = []


class QueueProgressStats(BaseModel):
    total: int = 0
    pending: int = 0
    locked: int = 0
    annotated: int = 0
    disputed: int = 0
    arbitrated: int = 0
    confirm_count: int = 0
    relabel_count: int = 0
    discard_count: int = 0
    confirm_rate: float = 0.0
    relabel_rate: float = 0.0
    discard_rate: float = 0.0


class AnnotationQueueResponse(BaseModel):
    id: int
    version_id: int
    name: str
    status: QueueStatus
    capacity: int
    review_mode: str
    num_reviewers: int
    lock_timeout_minutes: int
    created_by: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    target_version_id: Optional[int] = None
    priority_strategy: PriorityStrategy = PriorityStrategy.uncertainty
    webhook_url: Optional[str] = None
    webhook_thresholds: list = []
    triggered_thresholds: list = []
    progress: Optional[QueueProgressStats] = None

    class Config:
        from_attributes = True


class AnnotationSampleResponse(BaseModel):
    item_id: int
    sample_id: int
    text: str
    current_label: str
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    perplexity: Optional[float] = None
    uncertainty_score: float
    source_sample_id: Optional[int] = None


class ClaimTasksRequest(BaseModel):
    queue_id: int
    annotator_id: str
    batch_size: int = Field(default=10, ge=1, le=100)


class AnnotationSubmitItem(BaseModel):
    item_id: int
    decision: AnnotationDecision
    new_label: Optional[str] = None
    comment: Optional[str] = None


class SubmitAnnotationsRequest(BaseModel):
    queue_id: int
    annotator_id: str
    items: list[AnnotationSubmitItem]


class ReleaseLocksRequest(BaseModel):
    queue_id: int
    annotator_id: str
    item_ids: list[int] = []


class ArbitrateItem(BaseModel):
    item_id: int
    decision: AnnotationDecision
    new_label: Optional[str] = None
    comment: Optional[str] = None


class ArbitrateRequest(BaseModel):
    queue_id: int
    arbitrator_id: str
    items: list[ArbitrateItem]


class DisputedItemResponse(BaseModel):
    item_id: int
    sample_id: int
    text: str
    current_label: str
    records: list[dict]
    uncertainty_score: float


class ApplyQueueRequest(BaseModel):
    queue_id: int
    applied_by: str = ""


class AnnotatorStats(BaseModel):
    annotator_id: str
    total_annotated: int = 0
    confirm_count: int = 0
    relabel_count: int = 0
    discard_count: int = 0
    avg_annotation_seconds: Optional[float] = None
    agreement_rate: Optional[float] = None
    total_agreed: int = 0


class AnnotatorPerformanceResponse(BaseModel):
    annotator_id: str
    total_annotated: int = 0
    confirm_count: int = 0
    relabel_count: int = 0
    discard_count: int = 0
    avg_annotation_seconds: Optional[float] = None
    median_annotation_seconds: Optional[float] = None
    agreement_rate: Optional[float] = None
    total_agreed: int = 0
    decision_distribution: dict = {}


class BulkImportResult(BaseModel):
    total_records: int = 0
    imported_count: int = 0
    errors: list = []
    sample_ids_not_found: list = []
    invalid_decisions: list = []
    missing_new_labels: list = []


class RecommendedFilterConfigResponse(BaseModel):
    id: int
    version_id: int
    queue_id: Optional[int] = None
    source_config_name: str = "standard"
    ppl_multiplier: Optional[float] = None
    similarity_threshold: Optional[float] = None
    jaccard_threshold: Optional[float] = None
    label_confidence_threshold: Optional[float] = None
    adjustments: dict = {}
    reasoning: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConsistencyReport(BaseModel):
    queue_id: int
    cohens_kappa: float = 0.0
    kappa_level: str = ""
    warning: Optional[str] = None
    annotator_stats: list[AnnotatorStats] = []
    pairwise_kappa: Optional[dict] = None
