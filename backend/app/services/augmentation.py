import random
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    AugmentationTask, DatasetVersion, Sample, TaskStatus, SampleSource, SplitType
)
from ..config import MAX_AUGMENTATION_MULTIPLIER

logger = logging.getLogger(__name__)

_task_store: dict[int, "AugmentationTaskState"] = {}


class AugmentationTaskState:
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.paused = False
        self.cancelled = False
        self.processed = 0
        self.total = 0
        self.start_time: Optional[float] = None
        self.current_step_index = 0
        self.current_step_processed = 0
        self.current_step_total = 0


def get_task_state(task_id: int) -> AugmentationTaskState:
    if task_id not in _task_store:
        _task_store[task_id] = AugmentationTaskState(task_id)
    return _task_store[task_id]


def remove_task_state(task_id: int):
    _task_store.pop(task_id, None)


class BaseAugmenter(ABC):
    @abstractmethod
    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        pass


class SynonymReplaceAugmenter(BaseAugmenter):
    def __init__(self):
        self._nltk_ready = False
        self._stopwords = set()
        self._cn_synonym_dict: Optional[dict] = None

    def _ensure_nltk(self):
        if self._nltk_ready:
            return
        try:
            import nltk
            try:
                nltk.data.find("corpus/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            try:
                nltk.data.find("corpus/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words("english"))
            self._nltk_ready = True
        except Exception as e:
            logger.warning(f"NLTK init failed: {e}")
            self._nltk_ready = False

    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        language = params.get("language", "en")
        replace_ratio = params.get("replace_ratio", 0.1)

        try:
            if language == "zh":
                result = self._augment_chinese(text, replace_ratio)
            else:
                result = self._augment_english(text, replace_ratio)
            return [r for r in result if r.strip() and r.strip() != text.strip()]
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {e}")
            return []

    def _augment_english(self, text: str, replace_ratio: float) -> list[str]:
        try:
            from nltk.corpus import wordnet
        except Exception as e:
            logger.warning(f"NLTK WordNet not available: {e}")
            return []

        self._ensure_nltk()
        if not self._nltk_ready:
            logger.warning("NLTK not ready, skipping synonym replacement")
            return []

        words = text.split()
        if not words:
            return []

        non_stop_indices = [i for i, w in enumerate(words) if w.lower() not in self._stopwords and w.isalpha()]
        if not non_stop_indices:
            return []

        num_replace = max(1, int(len(non_stop_indices) * replace_ratio))
        replace_indices = random.sample(non_stop_indices, min(num_replace, len(non_stop_indices)))

        new_words = words.copy()
        changed = False
        for idx in replace_indices:
            synsets = wordnet.synsets(words[idx])
            synonyms = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != words[idx].lower() and name.isalpha():
                        synonyms.append(name)
            if synonyms:
                new_word = random.choice(synonyms)
                if new_word != new_words[idx]:
                    new_words[idx] = new_word
                    changed = True

        return [" ".join(new_words)] if changed else []

    def _augment_chinese(self, text: str, replace_ratio: float) -> list[str]:
        try:
            if self._cn_synonym_dict is None:
                self._cn_synonym_dict = self._load_chinese_synonyms()

            chars = list(text)
            replaceable = [i for i, c in enumerate(chars) if '\u4e00' <= c <= '\u9fff']
            if not replaceable:
                return []

            num_replace = max(1, int(len(replaceable) * replace_ratio))
            replace_indices = random.sample(replaceable, min(num_replace, len(replaceable)))

            new_chars = chars.copy()
            changed = False
            for idx in replace_indices:
                synonyms = self._cn_synonym_dict.get(chars[idx], [])
                if synonyms:
                    new_char = random.choice(synonyms)
                    if new_char != new_chars[idx]:
                        new_chars[idx] = new_char
                        changed = True

            return ["".join(new_chars)] if changed else []
        except Exception as e:
            logger.warning(f"Chinese synonym replacement failed: {e}")
            return []

    @staticmethod
    def _load_chinese_synonyms() -> dict:
        try:
            import json
            from pathlib import Path
            dict_path = Path(__file__).parent.parent.parent / "resources" / "cn_synonyms.json"
            if dict_path.exists():
                with open(dict_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load Chinese synonym dict: {e}")
        return {}


class RandomOpsAugmenter(BaseAugmenter):
    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        words = text.split()
        if len(words) < 2:
            return []

        n_ops = params.get("n_ops")
        if n_ops is None:
            n_ops = max(1, int(len(words) * 0.1))

        results = []
        seen = set()
        for _ in range(min(n_ops, 4)):
            op = random.choice(["insert", "swap", "delete"])
            new_words = words.copy()

            if op == "insert":
                self._random_insert(new_words)
            elif op == "swap":
                self._random_swap(new_words)
            elif op == "delete":
                delete_prob = params.get("delete_prob", 0.1)
                new_words = [w for w in new_words if random.random() > delete_prob]

            new_text = " ".join(new_words)
            if new_text and new_text != text and new_text not in seen:
                seen.add(new_text)
                results.append(new_text)

        return results

    @staticmethod
    def _random_insert(words: list[str]):
        if not words:
            return
        word = random.choice(words)
        synonyms = _get_simple_synonyms(word)
        if synonyms:
            syn = random.choice(synonyms)
            pos = random.randint(0, len(words))
            words.insert(pos, syn)

    @staticmethod
    def _random_swap(words: list[str]):
        if len(words) < 2:
            return
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]


def _get_simple_synonyms(word: str) -> list[str]:
    try:
        from nltk.corpus import wordnet
        synsets = wordnet.synsets(word)
        synonyms = []
        for syn in synsets[:2]:
            for lemma in syn.lemmas()[:2]:
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.append(name)
        return synonyms[:5]
    except Exception:
        return []


class BackTranslationAugmenter(BaseAugmenter):
    def __init__(self):
        self._models: dict = {}

    def _get_translation_models(self, src_lang: str, tgt_lang: str):
        key = f"{src_lang}-{tgt_lang}"
        if key not in self._models:
            try:
                from transformers import MarianMTModel, MarianTokenizer
                model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                self._models[key] = (tokenizer, model)
            except Exception as e:
                logger.error(f"Failed to load translation model {key}: {e}")
                raise
        return self._models[key]

    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        source_language = params.get("source_language", "en")
        pivot_language = params.get("pivot_language", "fr")
        num_variants = params.get("num_variants", 1)

        results = []
        for _ in range(num_variants):
            try:
                translated = self._translate(text, source_language, pivot_language)
                back_translated = self._translate(translated, pivot_language, source_language)
                if back_translated.strip() and back_translated.strip() != text.strip():
                    results.append(back_translated.strip())
            except Exception as e:
                logger.warning(f"Back-translation failed, skipping: {e}")
                continue

        return results

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        lang_map = {"en": "en", "fr": "fr", "de": "de", "zh": "zh", "ja": "ja"}
        src = lang_map.get(src_lang, src_lang)
        tgt = lang_map.get(tgt_lang, tgt_lang)

        tokenizer, model = self._get_translation_models(src, tgt)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=512, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


class ContextAugmentAugmenter(BaseAugmenter):
    def __init__(self):
        self._mlm_pipeline = None

    def _get_mlm(self, model_name: str = "bert-base-uncased"):
        if self._mlm_pipeline is None:
            try:
                from transformers import pipeline
                self._mlm_pipeline = pipeline("fill-mask", model=model_name, top_k=20)
            except Exception as e:
                logger.error(f"Failed to load MLM model: {e}")
                raise
        return self._mlm_pipeline

    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        mask_ratio = params.get("mask_ratio", 0.15)
        top_k = params.get("top_k", 5)
        num_variants = params.get("num_variants", 1)
        model_name = params.get("model_name", "bert-base-uncased")

        mlm = self._get_mlm(model_name)

        results = []
        for _ in range(num_variants):
            try:
                augmented = self._mask_and_fill(text, mask_ratio, top_k, mlm)
                if augmented and augmented != text:
                    results.append(augmented)
            except Exception as e:
                logger.warning(f"Context augmentation failed: {e}")
                continue

        return results

    @staticmethod
    def _mask_and_fill(text: str, mask_ratio: float, top_k: int, mlm) -> str:
        from transformers import AutoTokenizer
        tokenizer = mlm.tokenizer
        tokens = tokenizer.tokenize(text)
        if not tokens:
            return text

        num_mask = max(1, int(len(tokens) * mask_ratio))
        mask_indices = random.sample(range(len(tokens)), min(num_mask, len(tokens)))

        masked_tokens = tokens.copy()
        for idx in mask_indices:
            masked_tokens[idx] = "[MASK]"

        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        predictions = mlm(masked_text)

        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        if not predictions:
            return text

        filled_tokens = tokens.copy()
        pred_idx = 0
        for idx in mask_indices:
            if pred_idx < len(predictions):
                top_preds = predictions[pred_idx] if isinstance(predictions, list) else predictions
                if isinstance(top_preds, list):
                    candidates = top_preds[:top_k]
                    if candidates:
                        chosen = random.choice(candidates)
                        filled_tokens[idx] = chosen["token_str"].replace("##", "")
                pred_idx += 1

        return tokenizer.convert_tokens_to_string(filled_tokens)


class TemplateGenerateAugmenter(BaseAugmenter):
    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        template = params.get("template", "{label}类的例子: {text}")
        samples_per_seed = params.get("samples_per_seed", 3)
        pool = params.get("sample_pool", [])

        results = []
        for _ in range(samples_per_seed):
            if pool:
                other = random.choice(pool)
                other_text = other["text"]
                other_label = other["label"]
            else:
                other_text = text
                other_label = label

            new_text = template.replace("{label}", other_label).replace("{text}", other_text)
            results.append(new_text)

        return results


AUGMENTERS = {
    "synonym_replacement": SynonymReplaceAugmenter(),
    "random_ops": RandomOpsAugmenter(),
    "back_translation": BackTranslationAugmenter(),
    "context_augment": ContextAugmentAugmenter(),
    "template_generation": TemplateGenerateAugmenter(),
}


def validate_composite_strategy(steps: list[dict]) -> dict:
    if not steps:
        return {"valid": False, "reason": "Composite strategy must have at least one step"}

    strategies = [s["strategy"] for s in steps]

    for s in strategies:
        if s not in AUGMENTERS:
            return {"valid": False, "reason": f"Unknown strategy in steps: {s}"}

    if "back_translation" in strategies and strategies[0] != "back_translation":
        return {
            "valid": False,
            "reason": "Back-translation must be the first step in a composite strategy, "
                      "as it requires the complete original sentence as input."
        }

    for i in range(len(steps) - 1):
        if steps[i]["strategy"] == "random_ops" and steps[i + 1]["strategy"] == "context_augment":
            return {
                "valid": False,
                "reason": f"Context augmentation (MLM) cannot immediately follow random deletion at step {i + 1}. "
                          "Random deletion may produce incomplete sentences that affect BERT prediction quality. "
                          "Please add another strategy between them or reorder."
            }

    return {"valid": True}


async def execute_augmentation_task(
    session: AsyncSession,
    task_id: int,
) -> None:
    stmt = select(AugmentationTask).where(AugmentationTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        return

    if task.is_composite:
        await execute_composite_augmentation_task(session, task_id)
        return

    task.status = TaskStatus.running
    task.started_at = __import__("datetime").datetime.utcnow()
    await session.commit()

    state = get_task_state(task_id)
    state.start_time = time.time()

    try:
        stmt = select(Sample).where(
            Sample.version_id == task.source_version_id,
            Sample.is_filtered == False,
        )
        result = await session.execute(stmt)
        original_samples = result.scalars().all()

        if not original_samples:
            task.status = TaskStatus.completed
            task.completed_at = __import__("datetime").datetime.utcnow()
            await session.commit()
            return

        total = len(original_samples)
        state.total = total
        task.total_samples = total
        await session.commit()

        target_version = DatasetVersion(
            dataset_id=(await session.execute(select(DatasetVersion).where(DatasetVersion.id == task.source_version_id))).scalar_one().dataset_id,
            version_name=f"{task.strategy}_augmented",
            version_type="augmented",
            parent_version_id=task.source_version_id,
            split_ratios={},
        )
        session.add(target_version)
        await session.flush()
        task.target_version_id = target_version.id
        await session.commit()

        augmenter = AUGMENTERS.get(task.strategy)
        if not augmenter:
            task.status = TaskStatus.failed
            task.error_message = f"Unknown strategy: {task.strategy}"
            await session.commit()
            return

        source_samples_for_template = [
            {"text": s.text, "label": s.label} for s in original_samples
        ]
        strategy_params = dict(task.strategy_params)
        if task.strategy == "template_generation":
            strategy_params["sample_pool"] = source_samples_for_template

        for sample in original_samples:
            new_sample = Sample(
                version_id=target_version.id,
                text=sample.text,
                label=sample.label,
                split=sample.split,
                source=SampleSource.original,
                source_sample_id=sample.id,
            )
            session.add(new_sample)

        generated_count = 0
        max_total = int(total * task.augmentation_multiplier * MAX_AUGMENTATION_MULTIPLIER)
        max_additional = int(total * task.augmentation_multiplier)

        for i, sample in enumerate(original_samples):
            if state.cancelled:
                task.status = TaskStatus.failed
                task.error_message = "Cancelled by user"
                await session.commit()
                return

            while state.paused:
                await asyncio.sleep(1)

            try:
                new_texts = await augmenter.augment(sample.text, sample.label, strategy_params)
            except Exception as e:
                logger.warning(f"Augmentation failed for sample {sample.id}: {e}")
                new_texts = []

            for new_text in new_texts:
                if generated_count >= max_additional:
                    break
                new_sample = Sample(
                    version_id=target_version.id,
                    text=new_text,
                    label=sample.label,
                    split=sample.split,
                    source=SampleSource(task.strategy),
                    source_sample_id=sample.id,
                )
                session.add(new_sample)
                generated_count += 1

            state.processed = i + 1
            task.processed_samples = state.processed
            task.generated_samples = generated_count
            elapsed = time.time() - state.start_time
            if state.processed > 0:
                rate = elapsed / state.processed
                task.estimated_remaining_seconds = rate * (total - state.processed)
            await session.commit()

            if i % 10 == 0:
                await asyncio.sleep(0)

        class_dist = Counter()
        stmt_count = select(Sample).where(Sample.version_id == target_version.id)
        count_result = await session.execute(stmt_count)
        all_samples = count_result.scalars().all()
        for s in all_samples:
            class_dist[s.label] += 1

        target_version.total_samples = len(all_samples)
        target_version.class_distribution = dict(class_dist)
        target_version.split_ratios = (
            (await session.execute(select(DatasetVersion).where(DatasetVersion.id == task.source_version_id)))
            .scalar_one().split_ratios
        )

        task.status = TaskStatus.completed
        task.completed_at = __import__("datetime").datetime.utcnow()
        await session.commit()

    except Exception as e:
        logger.exception(f"Augmentation task {task_id} failed")
        task.status = TaskStatus.failed
        task.error_message = str(e)
        await session.commit()
    finally:
        remove_task_state(task_id)


async def pause_task(task_id: int) -> bool:
    state = get_task_state(task_id)
    state.paused = True
    return True


async def resume_task(task_id: int) -> bool:
    state = get_task_state(task_id)
    state.paused = False
    return True


async def cancel_task(task_id: int) -> bool:
    state = get_task_state(task_id)
    state.cancelled = True
    return True


async def preview_augmentation(
    session: AsyncSession,
    source_version_id: int,
    strategy: str,
    strategy_params: dict,
    sample_count: int = 5,
    timeout_seconds: int = 10,
) -> dict:
    stmt = select(Sample).where(
        Sample.version_id == source_version_id,
        Sample.is_filtered == False,
    )
    result = await session.execute(stmt)
    all_samples = result.scalars().all()

    if not all_samples:
        return {
            "strategy": strategy,
            "strategy_params": strategy_params,
            "samples": [],
            "total_count": 0,
            "success_count": 0,
            "timed_out_count": 0,
        }

    samples = random.sample(list(all_samples), min(sample_count, len(all_samples)))

    augmenter = AUGMENTERS.get(strategy)
    if not augmenter:
        raise ValueError(f"Unknown strategy: {strategy}")

    results = []
    success_count = 0
    timed_out_count = 0

    for sample in samples:
        sample_result = {
            "original_text": sample.text,
            "augmented_text": None,
            "timed_out": False,
            "error": None,
        }

        try:
            aug_coro = augmenter.augment(sample.text, sample.label, strategy_params)
            new_texts = await asyncio.wait_for(aug_coro, timeout=timeout_seconds)
            if new_texts:
                sample_result["augmented_text"] = new_texts[0]
                success_count += 1
        except asyncio.TimeoutError:
            sample_result["timed_out"] = True
            timed_out_count += 1
        except Exception as e:
            sample_result["error"] = str(e)

        results.append(sample_result)

    return {
        "strategy": strategy,
        "strategy_params": strategy_params,
        "samples": results,
        "total_count": len(results),
        "success_count": success_count,
        "timed_out_count": timed_out_count,
    }


async def execute_composite_augmentation_task(
    session: AsyncSession,
    task_id: int,
) -> None:
    from ..models.db_models import AugmentationStep

    stmt = select(AugmentationTask).where(AugmentationTask.id == task_id)
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        return

    task.status = TaskStatus.running
    task.started_at = __import__("datetime").datetime.utcnow()
    await session.commit()

    state = get_task_state(task_id)
    state.start_time = time.time()

    try:
        stmt_samples = select(Sample).where(
            Sample.version_id == task.source_version_id,
            Sample.is_filtered == False,
        )
        result_samples = await session.execute(stmt_samples)
        original_samples = result_samples.scalars().all()

        if not original_samples:
            task.status = TaskStatus.completed
            task.completed_at = __import__("datetime").datetime.utcnow()
            await session.commit()
            return

        total = len(original_samples)
        state.total = total
        task.total_samples = total
        await session.commit()

        target_version = DatasetVersion(
            dataset_id=(await session.execute(select(DatasetVersion).where(DatasetVersion.id == task.source_version_id))).scalar_one().dataset_id,
            version_name=f"composite_augmented",
            version_type="augmented",
            parent_version_id=task.source_version_id,
            split_ratios={},
        )
        session.add(target_version)
        await session.flush()
        task.target_version_id = target_version.id
        await session.commit()

        steps_stmt = select(AugmentationStep).where(
            AugmentationStep.task_id == task_id
        ).order_by(AugmentationStep.step_order)
        steps_result = await session.execute(steps_stmt)
        steps = steps_result.scalars().all()

        if not steps:
            task.status = TaskStatus.failed
            task.error_message = "No steps found for composite task"
            await session.commit()
            return

        num_steps = len(steps)
        step_stats = []

        for sample in original_samples:
            new_sample = Sample(
                version_id=target_version.id,
                text=sample.text,
                label=sample.label,
                split=sample.split,
                source=SampleSource.original,
                source_sample_id=sample.id,
            )
            session.add(new_sample)

        await session.flush()

        current_texts = [(sample.id, sample.text, sample.label, sample.split) for sample in original_samples]
        generated_count_per_step = [0] * num_steps

        for step_idx, step in enumerate(steps):
            state.current_step_index = step_idx
            state.current_step_processed = 0
            state.current_step_total = len(current_texts)
            task.current_step_index = step_idx
            step.input_count = len(current_texts)
            await session.commit()

            augmenter = AUGMENTERS.get(step.strategy)
            if not augmenter:
                task.status = TaskStatus.failed
                task.error_message = f"Unknown strategy in step {step_idx}: {step.strategy}"
                await session.commit()
                return

            step_params = dict(step.strategy_params)
            if step.strategy == "template_generation":
                source_samples_for_template = [
                    {"text": t[1], "label": t[2]} for t in current_texts
                ]
                step_params["sample_pool"] = source_samples_for_template

            next_texts = []
            success_count = 0
            skipped_count = 0

            for i, (orig_id, text, label, split) in enumerate(current_texts):
                if state.cancelled:
                    task.status = TaskStatus.failed
                    task.error_message = "Cancelled by user"
                    await session.commit()
                    return

                while state.paused:
                    await asyncio.sleep(1)

                try:
                    new_texts = await augmenter.augment(text, label, step_params)
                except Exception as e:
                    logger.warning(f"Augmentation step {step_idx} failed for sample {orig_id}: {e}")
                    new_texts = []

                if new_texts:
                    for new_text in new_texts[:1]:
                        next_texts.append((orig_id, new_text, label, split))
                        success_count += 1
                        generated_count_per_step[step_idx] += 1

                        if step_idx == num_steps - 1:
                            new_sample = Sample(
                                version_id=target_version.id,
                                text=new_text,
                                label=label,
                                split=split,
                                source=SampleSource(step.strategy),
                                source_sample_id=orig_id,
                            )
                            session.add(new_sample)
                else:
                    next_texts.append((orig_id, text, label, split))
                    skipped_count += 1

                state.current_step_processed = i + 1
                state.processed = int(
                    (step_idx + state.current_step_processed / state.current_step_total) / num_steps * state.total
                )
                task.processed_samples = state.processed
                task.generated_samples = sum(generated_count_per_step)

                elapsed = time.time() - state.start_time
                if state.processed > 0:
                    rate = elapsed / state.processed
                    task.estimated_remaining_seconds = rate * (state.total - state.processed)

                if i % 10 == 0:
                    await session.commit()
                    await asyncio.sleep(0)

            step.success_count = success_count
            step.skipped_count = skipped_count
            step_stats.append({
                "step_order": step.step_order,
                "strategy": step.strategy,
                "input_count": step.input_count,
                "success_count": success_count,
                "skipped_count": skipped_count,
            })
            task.step_stats = step_stats
            current_texts = next_texts
            await session.commit()

        class_dist = Counter()
        stmt_count = select(Sample).where(Sample.version_id == target_version.id)
        count_result = await session.execute(stmt_count)
        all_samples = count_result.scalars().all()
        for s in all_samples:
            class_dist[s.label] += 1

        target_version.total_samples = len(all_samples)
        target_version.class_distribution = dict(class_dist)
        target_version.split_ratios = (
            (await session.execute(select(DatasetVersion).where(DatasetVersion.id == task.source_version_id)))
            .scalar_one().split_ratios
        )

        task.status = TaskStatus.completed
        task.completed_at = __import__("datetime").datetime.utcnow()
        await session.commit()

    except Exception as e:
        logger.exception(f"Composite augmentation task {task_id} failed")
        task.status = TaskStatus.failed
        task.error_message = str(e)
        await session.commit()
    finally:
        remove_task_state(task_id)
