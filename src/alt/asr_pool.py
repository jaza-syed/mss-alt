# stdlib
import copy
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass, field
import logging
import os
import traceback

# third-party
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch
from tqdm import tqdm

# first-party
from .alt_types import TranscriptionTask, SongInfo, Segment
from alt import util

logger = logging.getLogger(__name__)


# thank u claude
class AsrPool:
    def __init__(
        self,
        num_gpus: int | None = None,
        model_name: str = "large-v2",
        batched: bool = False,
        model_type: str = "whisper",
        cached: bool = False,
        infer_dir: Path | None = None,
    ):
        """Initialize a pool of Whisper models across available GPUs"""

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        logger.info(
            f"Initializing WhisperPool with {num_gpus} models, using {model_name}"
        )

        self.models: list[BatchedInferencePipeline | WhisperModel] = []
        self.model_locks: list[threading.Lock] = []  # Lock for each model
        self.task_queue: Queue[TranscriptionTask] = Queue()
        self.result_dict: dict[str, tuple[list[Segment], TranscriptionTask]] = {}
        self.result_lock: threading.Lock = threading.Lock()
        self.pbar: tqdm | None = None
        self.batched: bool = batched
        self.cached: bool = cached
        self.infer_dir: Path | None = infer_dir
        if self.cached and not self.infer_dir:
            raise ValueError("If in cached mode, need infer dir")

        # Initialize models and their locks
        for gpu_id in range(num_gpus):
            model = WhisperModel(
                model_name, device="cuda", compute_type="float16", device_index=gpu_id
            )
            if self.batched:
                model = BatchedInferencePipeline(model=model)

            self.models.append(model)
            self.model_locks.append(threading.Lock())

        self.worker_threads = []

    def run_asr(self, asr: WhisperModel, task: TranscriptionTask) -> list[Segment]:
        if self.cached:
            assert self.infer_dir is not None
            pzfn = self.infer_dir / (task.uid + ".pz")
            if os.path.exists(pzfn):
                songinfo: SongInfo = util.read_pz(pzfn)
                assert songinfo.infer is not None
                assert songinfo.infer.whisper_result is not None
                return songinfo.infer.whisper_result
        segments, _info = asr.transcribe(str(task.audio_path), **task.transcribe_kwargs)
        return [Segment(start=s.start, end=s.end, text=s.text) for s in segments]

    def worker(self, worker_id: int):
        """Worker thread that processes tasks using a specific model"""
        while True:
            try:
                task: TranscriptionTask = self.task_queue.get(timeout=5)
            except Empty:
                break
            # Get exclusive access to the model
            with self.model_locks[worker_id]:
                try:
                    result = self.run_asr(self.models[worker_id], task)
                except Exception as e:
                    logger.fatal(f"Error in transcribe worker: {e}")
                    # Use instead of sys.exit as that works by throwing an exception
                    logger.fatal(traceback.format_exc())
                    os._exit(1)
                with self.result_lock:
                    self.result_dict[task.uid] = (result, task)
                    if self.pbar is not None:
                        self.pbar.update(1)
            # torch.cuda.empty_cache()
            self.task_queue.task_done()

    def transcribe_batch(
        self, tasks: list[TranscriptionTask]
    ) -> dict[str, tuple[list[Segment], TranscriptionTask]]:
        """
        Transcribe a batch of audio files using the model pool

        Args:
            tasks: list of tuples (audio_path, transcribe_kwargs) where transcribe_kwargs
                  contains the arguments for each transcription

        Returns:
            dictionary mapping audio files to their transcription results
        """
        if len({task.uid for task in tasks}) != len(tasks):
            raise ValueError("Duplicate task UIDs!")
        # Clear any previous results
        self.result_dict.clear()

        # Add all tasks to the queue
        for idx, task in enumerate(tasks):
            task.task_id = idx
            self.task_queue.put(task)
        self.pbar = tqdm(total=len(tasks))

        # Start worker threads (one per model)
        self.worker_threads = []
        for i in range(len(self.models)):
            thread = threading.Thread(target=self.worker, args=(i,))
            thread.start()
            self.worker_threads.append(thread)

        # Wait for all tasks to complete
        self.task_queue.join()

        # Wait for all threads to finish
        for thread in self.worker_threads:
            thread.join()
        if self.pbar is not None:
            self.pbar.close()

        # Prepare results in the original order
        return copy.deepcopy(self.result_dict)


if __name__ == "__main__":
    # Configure logger
    logger.basicConfig(level=logging.INFO)

    # Path to a sample audio file
    sample_audio_path = Path("./rsc/¡Óyeme_tiburón!_-_Corrientes_jam-alt_vocals.mp3")

    # Check if the sample audio file exists
    if not os.path.exists(sample_audio_path):
        raise FileNotFoundError(f"Sample audio file not found: {sample_audio_path}")

    # Create a WhisperPool instance
    whisper_pool = AsrPool(num_gpus=None, model_name="large-v2")

    # Define a batch of tasks (in this case, just one task)
    tasks = [
        TranscriptionTask(
            uid=f"task_{i}",
            audio_path=sample_audio_path,
            transcribe_kwargs={"language": "en", "beam_size": i % 4 + 1},
        )
        for i in range(1, torch.cuda.device_count() * 4)
    ]

    # Transcribe the batch
    results = whisper_pool.transcribe_batch(tasks)

    # Print the results
    for uid, transcription in results.items():
        print(f"Transcription for {uid}:")
        for segment in transcription:
            print(segment)
