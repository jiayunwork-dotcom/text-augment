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
            self._nltk_ready = True

    async def augment(self, text: str, label: str, params: dict) -> list[str]:
        language = params.get("language", "en")
        replace_ratio = params.get("replace_ratio", 0.1)

        if language == "zh":
            return self._augment_chinese(text, replace_ratio)
        return self._augment_english(text, replace_ratio)

    def _augment_english(self, text: str, replace_ratio: float) -> list[str]:
        self._ensure_nltk()
        from nltk.corpus import wordnet

        words = text.split()
        if not words:
            return [text]

        non_stop_indices = [i for i, w in enumerate(words) if w.lower() not in self._stopwords and w.isalpha()]
        if not non_stop_indices:
            return [text]

        num_replace = max(1, int(len(non_stop_indices) * replace_ratio))
        replace_indices = random.sample(non_stop_indices, min(num_replace, len(non_stop_indices)))

        new_words = words.copy()
        for idx in replace_indices:
            synsets = wordnet.synsets(words[idx])
            synonyms = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != words[idx].lower() and name.isalpha():
                        synonyms.append(name)
            if synonyms:
                new_words[idx] = random.choice(synonyms)

        return [" ".join(new_words)]

    def _augment_chinese(self, text: str, replace_ratio: float) -> list[str]:
        if self._cn_synonym_dict is None:
            self._cn_synonym_dict = self._load_chinese_synonyms()

        chars = list(text)
        replaceable = [i for i, c in enumerate(chars) if '\u4e00' <= c <= '\u9fff']
        if not replaceable:
            return [text]

        num_replace = max(1, int(len(replaceable) * replace_ratio))
        replace_indices = random.sample(replaceable, min(num_replace, len(replaceable)))

        new_chars = chars.copy()
        for idx in replace_indices:
            synonyms = self._cn_synonym_dict.get(chars[idx], [])
            if synonyms:
                new_chars[idx] = random.choice(synonyms)

        return ["".join(new_chars)]

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
            return [text]

        n_ops = params.get("n_ops")
        if n_ops is None:
            n_ops = max(1, int(len(words) * 0.1))

        results = []
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

            if new_words:
                results.append(" ".join(new_words))

        return results if results else [text]

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


async def execute_augmentation_task(
    session: AsyncSession,
    task_id: int,
) -> None:
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
        for s in original_samples:
            class_dist[s.label] += 1

        stmt_count = select(Sample).where(Sample.version_id == target_version.id)
        count_result = await session.execute(stmt_count)
        aug_samples = count_result.scalars().all()
        for s in aug_samples:
            class_dist[s.label] += 1

        target_version.total_samples = len(original_samples) + generated_count
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
