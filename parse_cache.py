"""
parse_cache.py

Provides a hash-based caching layer for document parsing.
Skips the expensive LLM call if the exact same CV or JD has been processed before.
"""
import os
import json
import hashlib
from typing import Type, Any
from pydantic import BaseModel

CACHE_DIR = ".cache"

def _get_hash(content: bytes) -> str:
    """Generates an MD5 hash of the file contents."""
    hasher = hashlib.md5()
    hasher.update(content)
    return hasher.hexdigest()

def _cv_cache_path(filepath: str) -> str:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(filepath, "rb") as f:
        file_hash = _get_hash(f.read())
    return os.path.join(CACHE_DIR, f"cv_{file_hash}.json")


def _jd_cache_path(filepath: str) -> str:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(filepath, "rb") as f:
        file_hash = _get_hash(f.read())
    return os.path.join(CACHE_DIR, f"jd_{file_hash}.json")


def is_cv_cached(filepath: str) -> bool:
    """Cheap pre-check (hash + file existence, no parsing) so a caller can
    report whether a CV parse will be instant or actually run the LLM,
    BEFORE running it — used by web_session.py's upload progress stream to
    give the GUI a truthful "loading from cache" vs "running local AI"
    message instead of a generic spinner."""
    return os.path.exists(_cv_cache_path(filepath))


def is_jd_cached(filepath: str) -> bool:
    return os.path.exists(_jd_cache_path(filepath))


def get_cached_or_parse_cv(filepath: str, parser_func: callable, response_model: Type[BaseModel]) -> Any:
    """Handles caching for CV PDFs (reads the file path)."""
    cache_path = _cv_cache_path(filepath)

    if os.path.exists(cache_path):
        print(f"   [Cache Hit] Instantly loading CV: {os.path.basename(filepath)}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return response_model.model_validate(json.load(f))

    print(f"   [Cache Miss] Running LLM parser for CV: {os.path.basename(filepath)}...")
    parsed_data = parser_func(filepath)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data.model_dump(), f, indent=2)
    return parsed_data

def _match_cache_path(cv_data: BaseModel, jd_data: BaseModel) -> str:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    combined = json.dumps(
        {"cv": cv_data.model_dump(), "jd": jd_data.model_dump()}, sort_keys=True
    ).encode("utf-8")
    return os.path.join(CACHE_DIR, f"match_{_get_hash(combined)}.json")


def check_match_cache(cv_data: BaseModel, jd_data: BaseModel, response_model: Type[BaseModel]) -> Any | None:
    """Returns the cached MatchResult for this exact (CV, JD) content pair, or
    None on a cache miss. Split out from get_cached_or_match so streaming
    callers (web_session.py) can check the cache before deciding whether to
    stream live matching or replay a cached result."""
    cache_path = _match_cache_path(cv_data, jd_data)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return response_model.model_validate(json.load(f))
    return None


def write_match_cache(cv_data: BaseModel, jd_data: BaseModel, match_result: BaseModel) -> None:
    cache_path = _match_cache_path(cv_data, jd_data)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(match_result.model_dump(), f, indent=2)


def get_cached_or_match(cv_data: BaseModel, jd_data: BaseModel, match_func: callable, response_model: Type[BaseModel]) -> Any:
    """
    Caches the baseline MatchResult itself, not just CV/JD parsing. match_cv_to_jd
    re-derives evidence retrieval + one LLM call per requirement from scratch every
    run (~5-15s x N requirements) even when parsing was already cache-hit — this
    closes that gap for repeat runs of the SAME (CV, JD) content pair. Hashed on
    the parsed data itself (not file paths), so it's correct even if filenames
    change but content doesn't.
    """
    cached = check_match_cache(cv_data, jd_data, response_model)
    if cached is not None:
        print("   [Cache Hit] Instantly loading baseline match result")
        return cached

    print("   [Cache Miss] Running baseline match...")
    match_result = match_func(cv_data, jd_data)
    write_match_cache(cv_data, jd_data, match_result)
    return match_result

def get_cached_or_parse_jd(filepath: str, parser_func: callable, response_model: Type[BaseModel]) -> Any:
    """Handles caching for JD text files (reads the text, passes string to parser)."""
    cache_path = _jd_cache_path(filepath)

    with open(filepath, "rb") as f:
        jd_text = f.read().decode("utf-8")

    if os.path.exists(cache_path):
        print(f"   [Cache Hit] Instantly loading JD: {os.path.basename(filepath)}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return response_model.model_validate(json.load(f))

    print(f"   [Cache Miss] Running LLM parser for JD: {os.path.basename(filepath)}...")
    parsed_data = parser_func(jd_text)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data.model_dump(), f, indent=2)
    return parsed_data