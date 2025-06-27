# stdlib
import copy
import threading
from queue import Queue, Empty
from pathlib import Path
import logging
import os
import traceback

# third-party
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch
from tqdm import tqdm

# first-party
from .alt_types import AsrTask, AsrSegment, ModelType
from alt import util

logger = logging.getLogger(__name__)


# thank u claude
class AsrPool:
    def __init__(
        self,
        num_gpus: int | None = None,
        model_name: str = "large-v2",
        batched: bool = False,
        model_type: ModelType = ModelType.WHISPER,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize a pool of Whisper models across available GPUs"""

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        logger.info(
            f"Initializing WhisperPool with {num_gpus} models, using {model_name}"
        )

        self.models: list[BatchedInferencePipeline | WhisperModel] = []
        self.model_locks: list[threading.Lock] = []  # Lock for each model
        self.task_queue: Queue[AsrTask] = Queue()
        self.result_dict: dict[str, list[AsrSegment]] = {}
        self.result_lock: threading.Lock = threading.Lock()
        self.pbar: tqdm | None = None
        self.batched: bool = batched
        self.cache_dir: Path | None = cache_dir

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

    def run_asr(
        self, asr: WhisperModel | BatchedInferencePipeline, task: AsrTask
    ) -> list[AsrSegment]:
        cache_fn = self.cache_dir / (task.uid + ".pz") if self.cache_dir else None
        if cache_fn and cache_fn.exists():
            return util.read_pz(cache_fn)

        segments, _info = asr.transcribe(str(task.audio_path), **task.transcribe_kwargs)
        result = [AsrSegment(start=s.start, end=s.end, text=s.text) for s in segments]

        if cache_fn:
            util.write_pz(cache_fn, result)
        return result

    def worker(self, worker_id: int) -> None:
        """Worker thread that processes tasks using a specific model"""
        while True:
            try:
                task: AsrTask = self.task_queue.get(timeout=5)
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
                    self.result_dict[task.uid] = result
                    if self.pbar is not None:
                        self.pbar.update(1)
            # torch.cuda.empty_cache()
            self.task_queue.task_done()

    def transcribe_batch(self, tasks: list[AsrTask]) -> dict[str, list[AsrSegment]]:
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
    logging.basicConfig(level=logging.INFO)

    # Path to a sample audio file
    sample_audio_path = Path("./rsc/¡Óyeme_tiburón!_-_Corrientes_jam-alt_vocals.mp3")

    # Check if the sample audio file exists
    if not os.path.exists(sample_audio_path):
        raise FileNotFoundError(f"Sample audio file not found: {sample_audio_path}")

    # Create a WhisperPool instance
    whisper_pool = AsrPool(num_gpus=None, model_name="large-v2")

    # Define a batch of tasks (in this case, just one task)
    tasks = [
        AsrTask(
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
