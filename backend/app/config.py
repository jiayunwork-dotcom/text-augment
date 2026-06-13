import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'text_augment.db'}")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "model_cache")))
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
MAX_AUGMENTATION_MULTIPLIER = 10
MIN_SAMPLES_PER_CLASS = 30
MAX_SELF_TRAINING_ITERATIONS = 5
SELF_TRAINING_EARLY_STOP_PATIENCE = 2
DEFAULT_PPL_THRESHOLD_MULTIPLIER = 3.0
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_JACCARD_THRESHOLD = 0.9
DEFAULT_LABEL_CONFIDENCE_THRESHOLD = 0.8

FILTER_PRESETS = {
    "loose": {
        "ppl_multiplier": 5.0,
        "similarity_threshold": 0.5,
        "jaccard_threshold": 0.95,
        "label_confidence_threshold": 0.9,
    },
    "standard": {
        "ppl_multiplier": 3.0,
        "similarity_threshold": 0.7,
        "jaccard_threshold": 0.9,
        "label_confidence_threshold": 0.8,
    },
    "strict": {
        "ppl_multiplier": 2.0,
        "similarity_threshold": 0.85,
        "jaccard_threshold": 0.8,
        "label_confidence_threshold": 0.7,
    },
}

BACK_TRANSLATION_PAIRS = {
    "en-de-en": {"source": "en", "pivot": "de", "source_family": "germanic", "pivot_family": "germanic"},
    "en-fr-en": {"source": "en", "pivot": "fr", "source_family": "germanic", "pivot_family": "romance"},
    "en-zh-en": {"source": "en", "pivot": "zh", "source_family": "germanic", "pivot_family": "sino_tibetan"},
    "en-ja-en": {"source": "en", "pivot": "ja", "source_family": "germanic", "pivot_family": "japonic"},
    "zh-en-zh": {"source": "zh", "pivot": "en", "source_family": "sino_tibetan", "pivot_family": "germanic"},
    "zh-ja-zh": {"source": "zh", "pivot": "ja", "source_family": "sino_tibetan", "pivot_family": "japonic"},
}

LANGUAGE_FAMILY_MAP = {
    "en": "germanic",
    "de": "germanic",
    "nl": "germanic",
    "sv": "germanic",
    "no": "germanic",
    "da": "germanic",
    "is": "germanic",
    "fr": "romance",
    "es": "romance",
    "it": "romance",
    "pt": "romance",
    "ro": "romance",
    "ca": "romance",
    "oc": "romance",
    "zh": "sino_tibetan",
    "bo": "sino_tibetan",
    "my": "sino_tibetan",
    "ja": "japonic",
    "ko": "koreanic",
    "ru": "slavic",
    "pl": "slavic",
    "cs": "slavic",
    "uk": "slavic",
    "bg": "slavic",
    "sr": "slavic",
    "ar": "semitic",
    "he": "semitic",
    "hi": "indo_aryan",
    "bn": "indo_aryan",
    "ta": "dravidian",
    "te": "dravidian",
    "tr": "turkic",
    "az": "turkic",
    "fi": "uralic",
    "hu": "uralic",
    "et": "uralic",
    "el": "hellenic",
    "la": "latinate",
    "ga": "celtic",
    "cy": "celtic",
}

SAME_FAMILY_BLACKLIST = {
    "germanic": {"germanic"},
    "romance": {"romance"},
    "sino_tibetan": {"sino_tibetan"},
    "japonic": {"japonic"},
    "koreanic": {"koreanic"},
    "slavic": {"slavic"},
    "semitic": {"semitic"},
    "indo_aryan": {"indo_aryan"},
    "dravidian": {"dravidian"},
    "turkic": {"turkic"},
    "uralic": {"uralic"},
    "hellenic": {"hellenic"},
    "celtic": {"celtic"},
    "latinate": {"romance"},
}

INVALID_PIVOT_SUGGESTIONS = {
    "en": "fr (Romance), es (Romance), zh (Sino-Tibetan), ja (Japonic), ru (Slavic), ar (Semitic)",
    "zh": "en (Germanic), fr (Romance), es (Romance), de (Germanic), ru (Slavic)",
    "de": "fr (Romance), es (Romance), zh (Sino-Tibetan), ja (Japonic), ru (Slavic)",
    "fr": "de (Germanic), zh (Sino-Tibetan), ja (Japonic), ru (Slavic), ar (Semitic)",
    "es": "de (Germanic), zh (Sino-Tibetan), ja (Japonic), ru (Slavic), ar (Semitic)",
    "ja": "en (Germanic), fr (Romance), es (Romance), de (Germanic), ru (Slavic)",
    "default": "a language from a different language family (e.g., English/Germanic, French/Romance, Chinese/Sino-Tibetan)",
}
