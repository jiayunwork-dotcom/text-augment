import logging
from collections import Counter

from ..config import (
    MAX_AUGMENTATION_MULTIPLIER,
    BACK_TRANSLATION_PAIRS,
    SAME_FAMILY_BLACKLIST,
    MIN_SAMPLES_PER_CLASS,
)
from ..models.db_models import Sample, SampleSource, AugmentationTask

logger = logging.getLogger(__name__)


def validate_augmentation_multiplier(
    original_count: int,
    augmentation_multiplier: float,
    generated_count: int,
) -> dict:
    max_allowed = original_count * MAX_AUGMENTATION_MULTIPLIER
    if generated_count > max_allowed:
        logger.warning(
            f"Augmented samples ({generated_count}) exceed {MAX_AUGMENTATION_MULTIPLIER}x "
            f"original ({original_count}). Truncating."
        )
        return {
            "valid": False,
            "max_allowed": max_allowed,
            "actual": generated_count,
            "action": "truncate_proportionally",
        }
    return {"valid": True}


def validate_back_translation_pair(source_lang: str, pivot_lang: str) -> dict:
    for pair_name, pair_info in BACK_TRANSLATION_PAIRS.items():
        if pair_info["source"] == source_lang and pair_info["pivot"] == pivot_lang:
            source_family = pair_info["source_family"]
            pivot_family = pair_info["pivot_family"]
            blacklisted = SAME_FAMILY_BLACKLIST.get(source_family, set())
            if pivot_family in blacklisted:
                return {
                    "valid": False,
                    "reason": f"Source language ({source_lang}) and pivot language ({pivot_lang}) "
                              f"belong to the same language family ({source_family}). "
                              f"Please choose a pivot from a different language family.",
                    "suggestion": _suggest_pivot(source_lang),
                }
            return {"valid": True}

    return {
        "valid": True,
        "warning": f"Language pair ({source_lang}-{pivot_lang}) not in predefined pairs. "
                   f"Proceeding without family check.",
    }


def _suggest_pivot(source_lang: str) -> str:
    suggestions = {
        "en": "fr (Romance) or zh (Sino-Tibetan) or ja (Japonic)",
        "zh": "en (Germanic) or ja (Japonic)",
        "de": "fr (Romance) or zh (Sino-Tibetan)",
        "fr": "en (Germanic) or zh (Sino-Tibetan)",
    }
    return suggestions.get(source_lang, "Choose a language from a different language family")


def validate_split_consistency(
    original_sample_id: int,
    original_split: str,
    augmented_split: str,
) -> dict:
    if original_split != augmented_split:
        return {
            "valid": False,
            "reason": f"Data leak detected: original sample (split={original_split}) "
                      f"produced augmented sample in different split ({augmented_split})",
        }
    return {"valid": True}


def check_filter_mandatory(is_filtering_skipped: bool) -> dict:
    if is_filtering_skipped:
        return {
            "valid": False,
            "reason": "Quality filtering is mandatory before training. Cannot skip.",
        }
    return {"valid": True}


def check_class_minimum_samples(
    class_distribution: dict[str, int],
    source_map: dict[str, list[SampleSource]] = None,
) -> dict:
    undersampled = {}
    for label, count in class_distribution.items():
        if count < MIN_SAMPLES_PER_CLASS:
            undersampled[label] = {
                "current": count,
                "minimum": MIN_SAMPLES_PER_CLASS,
                "action": "oversample",
            }
    if undersampled:
        return {
            "needs_oversampling": True,
            "undersampled_classes": undersampled,
        }
    return {"needs_oversampling": False, "undersampled_classes": {}}


def truncate_augmented_samples(
    original_samples: list[dict],
    augmented_samples: list[dict],
    max_multiplier: float = MAX_AUGMENTATION_MULTIPLIER,
) -> list[dict]:
    max_total = int(len(original_samples) * max_multiplier)
    current_total = len(augmented_samples)

    if current_total <= max_total:
        return augmented_samples

    label_counts_original = Counter(s["label"] for s in original_samples)
    label_counts_augmented = Counter(s["label"] for s in augmented_samples)

    ratio_per_label = {}
    for label in label_counts_augmented:
        orig_count = label_counts_original.get(label, 1)
        aug_count = label_counts_augmented[label]
        ratio_per_label[label] = aug_count / max(orig_count, 1)

    target_per_label = {}
    for label in label_counts_augmented:
        orig_count = label_counts_original.get(label, 1)
        target_per_label[label] = int(orig_count * max_multiplier)

    truncated = []
    label_buckets: dict[str, list] = {}
    for s in augmented_samples:
        label = s["label"]
        if label not in label_buckets:
            label_buckets[label] = []
        label_buckets[label].append(s)

    for label, bucket in label_buckets.items():
        target = target_per_label.get(label, len(bucket))
        truncated.extend(bucket[:target])

    return truncated
