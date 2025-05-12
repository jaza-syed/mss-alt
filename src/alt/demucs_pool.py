# stdlib
import copy
import threading
from queue import Queue, Empty
from typing import List, Dict, Any
from pathlib import Path
import dataclasses
from dataclasses import dataclass
import torch
import logging
import os
import sys
import traceback

# third-party
import demucs.pretrained as pretrained
import demucs.separate as separate
import demucs.audio as audio
import demucs.apply as apply
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SeparationTask:
    """Container for a separation task with its arguments"""

    uid: str
    audio_path: Path
    output_dir: str
    apply_kwargs: Dict[str, Any]
    task_id: int | None = None


# thank u claude
class DemucsPool:
    def __init__(
        self,
        model_name: str = "mdx_extra",
        overwrite: bool = False,
        segment: int | None = None,
        cpu: bool = False,
    ):
        """Initialize a pool of Whisper models across available GPUs"""
        logger.info(f"Initializing DemucsPool with {model_name}")
        self.cpu : bool = cpu or torch.cuda.device_count() == 0
        self.num_workers : int = 0
        if self.cpu:
            self.num_workers = os.getenv("POOL_WORKERS", 2)
            logger.info(f"Using CPU with {self.num_workers} workers")
        else:
            self.num_workers = torch.cuda.device_count()
            logger.info(f"Using GPU with {self.num_workers} workers")
        self.models: List[apply.Model | apply.BagOfModels] = []
        self.model_locks: List[threading.Lock] = []  # Lock for each model
        self.task_queue: Queue[SeparationTask] = Queue()
        self.result_dict: dict[str, dict[str, str]] = {}
        self.result_lock: threading.Lock = threading.Lock()
        self.pbar: tqdm | None = None
        self.overwrite: bool = overwrite
        # Initialize models and their locks

        for worker_id in range(self.num_workers):
            model = pretrained.get_model(name=model_name)
            if segment:
                if isinstance(model, apply.BagOfModels):
                    for sub in model.models:
                        sub.segment = segment  # type: ignore
                else:
                    model.segment = segment
            model.to(self.get_device(worker_id))
            model.eval()
            self.models.append(model)
            self.model_locks.append(threading.Lock())

        self.worker_threads = []

    def get_device(self, worker_index: int) -> str:
        """Get the device string for a specific worker index"""
        return "cpu" if self.cpu else f"cuda:{worker_index}"

    def worker(self, worker_id: int):
        """Worker thread that processes tasks using a specific model"""
        while True:
            try:
                task: SeparationTask = self.task_queue.get(timeout=5)
            except Empty:
                break
            # Get exclusive access to the model
            with self.model_locks[worker_id]:
                model = self.models[worker_id]
                completion_file = os.path.join(task.output_dir, "complete")
                vocals_fn = os.path.join(task.output_dir, "vocals.mp3")

                def complete():
                    logger.debug(f"Separation task complete: {task.uid}, continue")
                    with self.result_lock:
                        self.result_dict[task.uid] = {"vocals": vocals_fn}
                        if self.pbar is not None:
                            self.pbar.update(1)
                    self.task_queue.task_done()

                # Actually run stuff
                if os.path.exists(completion_file) and not self.overwrite:
                    complete()
                    continue
                try:
                    if os.path.exists(completion_file):
                        os.remove(completion_file)
                    wav = separate.load_track(
                        task.audio_path, model.audio_channels, model.samplerate
                    )
                    ref = wav.mean(0)
                    wav = (wav - ref.mean()) / ref.std()
                    # Apply the model to extract sources
                    sources = apply.apply_model(
                        model, wav[None], device=self.get_device(worker_id), **task.apply_kwargs
                    )
                    sources = sources * ref.std() + ref.mean()
                    # Prepare saving arguments for ffmpeg (save as 320 kbps mp3)
                    save_kwargs = {
                        "samplerate": model.samplerate,
                        "bitrate": 320,
                        "clip": "clamp",
                        "as_float": True,
                        "bits_per_sample": 16,
                    }
                    # Save the extracted vocals
                    os.makedirs(task.output_dir, exist_ok=True)
                    audio.save_audio(
                        sources[0, model.sources.index("vocals")],
                        vocals_fn,
                        **save_kwargs,
                    )
                    with open(completion_file, "w") as f:
                        f.write("complete")

                except Exception as e:
                    logger.fatal(f"Error in transcribe worker for {task.uid}: {e}")
                    # Capture full stack trace
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    # Get all stack frames including assertion location
                    stack_trace = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    logger.fatal(stack_trace)

                    # Use instead of sys.exit as that works by throwing an exception
                    os._exit(1)
                complete()

    def separate_batch(self, tasks: List[SeparationTask]) -> Dict[str, Any]:
        """
        Transcribe a batch of audio files using the model pool

        Args:
            tasks: List of tuples (audio_path, transcribe_kwargs) where transcribe_kwargs
                  contains the arguments for each transcription

        Returns:
            Dictionary mapping audio files to their transcription results
        """
        if len({task.uid for task in tasks}) != len(tasks):
            logger.error(f"Num task uids: {len({task.uid for task in tasks})}")
            logger.error(f"Num tasks {len(tasks)}")
            raise ValueError("Duplicate task UIDs!")
        # Clear any previous results
        self.result_dict.clear()

        # Add all tasks to the queue
        for idx, task in enumerate(tasks):
            task = SeparationTask(**dataclasses.asdict(task))
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

    def cleanup(self):
        """Cleanup resources by removing models from GPU"""
        for model in self.models:
            model.to("cpu")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

    # Path to a sample audio file
    sample_audio_path = Path("./rsc/¡Óyeme_tiburón!_-_Corrientes_jam-alt_vocals.mp3")
    output_path = "./tmp/separate"
    os.makedirs(output_path, exist_ok=True)

    # Check if the sample audio file exists
    if not os.path.exists(sample_audio_path):
        raise FileNotFoundError(f"Sample audio file not found: {sample_audio_path}")

    # Create a WhisperPool instance
    demucs_pool = DemucsPool(num_gpus=None, model_name="mdx_extra")

    # Define a batch of tasks (in this case, just one task)
    tasks = [
        SeparationTask(
            uid=f"task_{i}",
            audio_path=sample_audio_path,
            output_dir=output_path,
            apply_kwargs={},
        )
        for i in range(torch.cuda.device_count() * 2)
    ]
    for idx, task in enumerate(tasks):
        task.output_dir = os.path.join(task.output_dir, str(idx))
        os.makedirs(task.output_dir, exist_ok=True)

    # Transcribe the batch
    results = demucs_pool.separate_batch(tasks)

    # Print the results
    for uid, stem_fns in results.items():
        print(f"Stems for {uid}: {stem_fns}")
