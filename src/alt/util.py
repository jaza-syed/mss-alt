# stdlib
import glob
import gzip
import json
import logging
import os
import pickle
import sys
from typing import Any
from pathlib import Path

# first-party
from . import alt_types
from .render import HtmlSummary

logger = logging.getLogger(__name__)


def data_root() -> Path:
    path = os.getenv("DATA_ROOT", None)
    if path is None:
        raise RuntimeError("DATA_ROOT must be set")
    return Path(path)


def read_songinfo(filename: str | Path) -> alt_types.SongInfo:
    with open(filename, "r", encoding="utf-8") as fin:
        return alt_types.SongInfo.schema().loads(fin.read())  # type: ignore


def read_config(filename: str | Path) -> alt_types.PipelineConfig:
    with open(filename, "r", encoding="utf-8") as fin:
        return alt_types.PipelineConfig.schema().loads(fin.read())  # type: ignore


def read_eval(filename: str | Path) -> HtmlSummary:
    with open(filename, "r", encoding="utf-8") as fin:
        return HtmlSummary.schema().loads(fin.read())  # type: ignore


def read_summary(filename: str | Path) -> alt_types.Summary:
    with open(filename, "r", encoding="utf-8") as fin:
        return alt_types.Summary.schema().loads(fin.read())  # type: ignore


def write_json_dataclass(obj: Any, filename: str | Path):
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write(obj.to_json(ensure_ascii=False, indent=2, sort_keys=True))


def readjson(filename: str | Path) -> Any:
    with open(filename, "r", encoding="utf-8") as fin:
        return json.loads(fin.read())


def writejson(obj: Any, filename: str | Path):
    with open(filename, "w", encoding="utf-8") as fout:
        return fout.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def find_cfg(cfg: str, cfg_dir: str) -> str | None:
    found_cfg = None
    for fn in glob.glob(os.path.join(cfg_dir, "*.json")):
        if fn.endswith(cfg + ".json"):
            if found_cfg is None:
                found_cfg = fn
            else:
                raise ValueError(f"Multiple configs with base {cfg}")
    if found_cfg is None:
        logger.error(f"Config '{cfg}' not found in '{cfg_dir}'")
        sys.exit(1)
    return found_cfg


def make_pipline_dir(cfg: str, stage: str) -> str:
    path = os.path.join("build", cfg, stage)
    os.makedirs(path)
    return path


def data_path(relative_path: str) -> str | None:
    data_root = os.getenv("DATA_ROOT", None)
    if data_root is None:
        return None
    return os.path.join(data_root, relative_path)


# read gzip'd pickle
def read_pz(pzfn) -> Any:
    with gzip.open(pzfn, "r") as fin:
        return pickle.load(fin)  # type: ignore


def write_pz(pzfn: Path, obj: Any):
    with gzip.open(pzfn, "wb") as fo:
        pickle.dump(obj, fo)


def default_log_config():
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
    logger.setLevel(level=os.getenv("LOGLEVEL", "INFO"))
    # Prevent overly verbose info logger
    logging.getLogger("faster_whisper").setLevel(logging.WARN)
    # logging.getLogger("espnet").setLevel(logging.WARN)
