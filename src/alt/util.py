# STDLIB
import gzip
from io import TextIOWrapper
import logging
import os
from pathlib import Path
import pickle
import subprocess
from typing import TypeVar, Generic, Iterator, Optional, Any
from typing_extensions import Self

# THIRDPARTY
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).parents[2]


# read gzip'd pickle
def read_pz(pzfn) -> Any:
    with gzip.open(pzfn, "r") as fin:
        return pickle.load(fin)  # type: ignore


def write_pz(pzfn: Path, obj: Any):
    with gzip.open(pzfn, "wb") as fo:
        pickle.dump(obj, fo)


def default_log_config() -> None:
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
    logger.setLevel(level=os.getenv("LOGLEVEL", "INFO"))
    # Prevent overly verbose info logger
    logging.getLogger("faster_whisper").setLevel(logging.WARN)
    # logging.getLogger("espnet").setLevel(logging.WARN)


T = TypeVar("T")


class DataFile(Generic[T]):
    """
    File-like class to store Pydantic dataclasses as gzipped JSONL files.
    """

    def __init__(
        self,
        filename: str | Path,
        data_type: type[T],
        mode: str,
    ) -> None:
        if mode not in ["r", "w"]:
            raise ValueError(f"{mode} must be 'r', 'w'")
        self.type_adapter: TypeAdapter[T] = TypeAdapter(data_type)
        self.filename: str | Path = filename
        self.mode: str = mode[0]
        self.file_handler: Optional[TextIOWrapper] = None

    def open(self) -> None:
        self.file_handler = gzip.open(self.filename, f"{self.mode}t")  # type: ignore

    def close(self) -> None:
        if self.file_handler is not None:
            self.file_handler.close()
            self.file_handler = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[T]:
        # Open if not open already
        if self.file_handler is None:
            with self as f:
                for data in f:
                    yield data
        else:
            for row in self.file_handler:
                yield self.type_adapter.validate_json(row)

    def write(self, data: T) -> None:
        # write data one at a time
        # TODO: ValueError shouldn't be necessary due to type checks
        if not isinstance(data, self.type_adapter._type):
            raise ValueError(
                f"Type of input data {data.__class__} doesn't match file's {self.type_adapter}"
            )
        assert self.file_handler is not None
        self.file_handler.write(self.type_adapter.dump_json(data).decode("utf-8"))
        self.file_handler.write("\n")

    def write_all(self, data: Iterator[T]) -> None:
        for datum in data:
            self.write(datum)


def model_dump_json(path: Path, obj: Any, indent=4, **kwargs) -> int:
    with open(path, "wt") as f:
        adapter = TypeAdapter(type(obj))
        return f.write(adapter.dump_json(obj, indent=indent, **kwargs).decode("utf-8"))


def model_read_json(path: Path, model_cls: type[T]) -> T:
    with open(path, "rt") as f:
        return TypeAdapter(model_cls).validate_json(f.read())


def read_datafile(filename: str | Path, data_type: type[T]) -> list[T]:
    with DataFile(filename, data_type, "r") as df:
        return list(df)


def assert_git_clean() -> None:
    status_text = subprocess.check_output(
        "git status --porcelain", shell=True, encoding="utf8"
    ).strip()
    assert len(status_text) == 0, (
        "This script will not proceed until your worktree is completely "
        + "clean (unstaged and staged files)."
    )


def data_root() -> Path:
    path = os.getenv("DATA_ROOT", None)
    if path is None:
        raise RuntimeError("Environment variable DATA_ROOT must be set")
    return Path(path)


def output_dir() -> Path:
    path = os.getenv("OUTPUT_DIR", None)
    if path is None:
        raise RuntimeError("Environment variable OUTPUT_DIR must be set")
    return Path(path)


def separation_dir() -> Path:
    return output_dir() / "sep"


def model_dir() -> Path:
    return output_dir() / "models"


def render_dir() -> Path:
    return output_dir() / "render"


def infer_dir() -> Path:
    return output_dir() / "infer"


def extract_dir() -> Path:
    return output_dir() / "extract"


def share_dir() -> Path:
    return output_dir() / "share"
