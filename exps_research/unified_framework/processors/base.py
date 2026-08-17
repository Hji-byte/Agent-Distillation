"""
Base experiment processor module
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
from queue import Empty
from tqdm import tqdm
from copy import deepcopy
import pickle
import time
import os
import multiprocessing as mp
import traceback

from .cost_tracker import CostTracker
from exps_research.unified_framework import setup_model
from exps_research.unified_framework.utils import append_result

# Global counter for multiprocessing progress tracking
_progress_counter = None

# Top-level function for process pool to avoid pickling issues
def process_entry_in_process(
        entry: Dict, 
        worker_id: int, 
        model_kwargs: Dict[str, Any], 
        use_local_model: bool, 
        verbose_worker: bool, 
        processor_class_module: str, 
        processor_class_name: str,
        use_single_endpoint: bool = False,
        **kwargs
        ):
    """
    Top-level function to process a single entry in a separate process
    
    Args:
        entry: The entry to process
        worker_id: ID of the worker
        model_kwargs: Model configuration parameters
        use_local_model: Whether to use local model
        verbose_worker: Whether this worker should show verbose output
        processor_class_module: The module containing the processor class
        processor_class_name: The name of the processor class
        use_single_endpoint: Whether to use a single API endpoint for all workers
        **kwargs: Additional parameters to pass to process_entry
        
    Returns:
        Processed result
    """
    # Dynamically import the processor class
    import importlib
    module = importlib.import_module(processor_class_module)
    processor_class = getattr(module, processor_class_name)
    
    # Make a deep copy of model_kwargs to avoid modifying the original
    model_kwargs_copy = deepcopy(model_kwargs)
    
    # Configure model parameters for this worker
    if use_local_model:
        model_kwargs_copy["local_device_id"] = str(worker_id)
    elif model_kwargs_copy.get("model_type") == "vllm":
        # Only modify API base if it's not explicitly set
        if use_single_endpoint:
            model_kwargs_copy["api_base"] = "http://0.0.0.0:8000/v1"
        else:
            model_kwargs_copy["api_base"] = f"http://0.0.0.0:{8000 + worker_id}/v1"
    
    # Create a temporary processor instance, then let it construct the model.
    # Besides avoiding duplicated setup logic, this keeps isolated workers
    # testable with lightweight processor-specific model factories.
    processor = processor_class(model_kwargs_copy, **kwargs)
    model = processor.create_model(worker_id, use_local_model, use_single_endpoint)
    
    # Process the entry
    result = processor.process_entry(
        entry, 
        model,
        verbose_worker=verbose_worker,
        **kwargs
    )
    
    return result


def process_entry_in_isolated_process(result_queue, *args, **kwargs):
    """Run one entry in a disposable process and report its outcome."""
    try:
        result = process_entry_in_process(*args, **kwargs)
        result_queue.put(("result", result))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        result_queue.close()
        result_queue.join_thread()


def _isolated_failure_result(
    entry: Dict[str, Any],
    model_id: str,
    *,
    state: str,
    message: str,
    timeout_seconds: Optional[float] = None,
    child_traceback: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an observable failure row that resume logic will retry."""
    source_metadata = {
        key: entry[key]
        for key in ("id", "dataset_name", "split", "level", "type")
        if key in entry
    }
    metadata = {"state": state, "success": False, "source": source_metadata}
    if timeout_seconds is not None:
        metadata["timeout_seconds"] = timeout_seconds
    if child_traceback:
        metadata["child_traceback"] = child_traceback

    return {
        "model_id": model_id,
        "question": entry.get("question"),
        "generated_answer": None,
        "true_answer": entry.get("answer"),
        "error": message,
        "log_data": {
            "schema_version": "isolated-agent-failure-v1",
            "trajectory_steps": [],
            "metadata": metadata,
        },
        "input_tokens": 0,
        "output_tokens": 0,
        "selected_input_tokens": 0,
        "selected_output_tokens": 0,
    }


class ExperimentProcessor(ABC):
    """
    Base class for experiment processors
    
    This abstract class defines the interface and common functionality
    for all experiment processors. Concrete implementations should inherit
    from this class and implement the process_entry method.
    """
    
    def __init__(self, model_kwargs: Dict[str, Any], **kwargs):
        """
        Initialize experiment processor
        
        Args:
            model_kwargs: Model configuration parameters
            **kwargs: Additional experiment-specific parameters
        """
        self.model_kwargs = model_kwargs
        self.cost_tracker = CostTracker()
        self.track_cost = kwargs.get('track_cost', False)
        self.verbose = kwargs.get('verbose', False)
        
        # Set up cost tracking if enabled
        if self.track_cost:
            self.cost_tracker.reset(kwargs.get('cost_threshold'))
    
    @abstractmethod
    def process_entry(self, entry: Dict, model, **kwargs) -> Dict:
        """
        Process a single experiment entry
        
        Args:
            entry: Dictionary containing a question/problem
            model: Model instance to use
            **kwargs: Additional experiment-specific parameters
            
        Returns:
            Result dictionary with generated answer and metadata
        """
        pass
    
    def create_model(self, worker_id: int = 0, use_local_model: bool = False, use_single_endpoint: bool = False) -> Any:
        """
        Create a model instance for this worker
        
        Args:
            worker_id: ID of the worker thread/process
            use_local_model: Whether to use a local model
            use_single_endpoint: Whether to use a single API endpoint for all workers
            
        Returns:
            Model instance
        """
        model_kwargs = deepcopy(self.model_kwargs)
        
        if use_local_model:
            model_kwargs["local_device_id"] = str(worker_id)
        elif model_kwargs.get("model_type") == "vllm":
            # Only modify API base if it's not explicitly set
            if not model_kwargs.get('api_base'):
                if use_single_endpoint:
                    model_kwargs["api_base"] = "http://0.0.0.0:8000/v1"
                else:
                    model_kwargs["api_base"] = f"http://0.0.0.0:{8000 + worker_id}/v1"
        
        return setup_model(**model_kwargs)
    
    def create_models(self, max_workers: int, use_local_model: bool = False, use_single_endpoint: bool = False) -> List:
        """
        Create model instances for all workers
        
        Args:
            max_workers: Number of worker threads/processes
            use_local_model: Whether to use local models
            use_single_endpoint: Whether to use a single API endpoint for all workers
            
        Returns:
            List of model instances
        """
        return [self.create_model(i, use_local_model, use_single_endpoint) for i in range(max_workers)]
    
    def process_dataset(
        self,
        entries: List[Dict],
        output_file: Optional[str] = None,
        max_workers: int = 1,
        debug: bool = False,
        use_local_model: bool = False,
        use_process_pool: bool = True,  # Default to process pool for reliable timeouts
        use_single_endpoint: bool = False,  # Use a single API endpoint for all workers
        isolate_agent_processes: bool = False,
        question_timeout_seconds: float = 300,
        **kwargs
    ) -> List[Dict]:
        """
        Process a dataset of entries
        
        Args:
            entries: List of dataset entries
            output_file: Path to output file
            max_workers: Maximum number of concurrent workers
            debug: Whether to run in debug mode
            use_local_model: Whether to use local models
            use_process_pool: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
                              ProcessPoolExecutor is recommended for reliable timeouts in Python code execution
            use_single_endpoint: Whether to use a single API endpoint (port 8000) for all workers
            isolate_agent_processes: Run every entry in a disposable child process
            question_timeout_seconds: Hard wall-clock limit for one isolated entry
            **kwargs: Additional experiment-specific parameters
            
        Returns:
            List of processed results
        """
        results = []
        
        # Limit entries in debug mode
        if debug:
            entries = entries[:10]
            # max_workers = 1

        if isolate_agent_processes:
            if question_timeout_seconds <= 0:
                raise ValueError("question_timeout_seconds must be greater than zero")
            return self._process_dataset_isolated(
                entries,
                output_file=output_file,
                max_workers=max_workers,
                use_local_model=use_local_model,
                use_single_endpoint=use_single_endpoint,
                question_timeout_seconds=question_timeout_seconds,
                **kwargs,
            )
        
        # Process sequentially if single worker or debug mode
        if max_workers <= 1:
            model = self.create_model(0, use_local_model, use_single_endpoint)
            
            print(f"Processing {len(entries)} questions sequentially")
            for entry in tqdm(entries, desc=f"Processing questions"):
                if self.cost_tracker.stop_requested:
                    print(f"\nCost threshold reached. Stopping execution.")
                    break
                
                # In sequential mode, we can always show verbose output if enabled
                result = self.process_entry(entry, model, verbose_worker=True, **kwargs)
                
                if result:
                    results.append(result)
                    if output_file:
                        append_result(result, output_file)
                    if self.track_cost and "cost" in result:
                        self.cost_tracker.update_cost(result["cost"])
        else:
            # Parallel processing
            pool_type = "process" if use_process_pool else "thread"
            endpoint_type = "single" if use_single_endpoint else "multiple"
            print(f"Processing {len(entries)} questions with {max_workers} workers using {pool_type} pool and {endpoint_type} endpoint(s)")
            
            if use_process_pool:
                # Process-based parallelism (better for timeouts)
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = []
                    
                    # Get the processor class module and name for dynamic import
                    processor_class_module = self.__class__.__module__
                    processor_class_name = self.__class__.__name__
                    
                    print(f"Submitting {len(entries)} tasks to process pool...")
                    # Submit all tasks
                    for i, entry in enumerate(entries):
                        worker_id = i % max_workers
                        futures.append(executor.submit(
                            process_entry_in_process,
                            entry,
                            worker_id,
                            self.model_kwargs,
                            use_local_model,
                            worker_id == 0,  # Only first worker shows output
                            processor_class_module,
                            processor_class_name,
                            use_single_endpoint,
                            **{k: v for k, v in kwargs.items() if k != 'self'}  # Filter out self reference
                        ))
                    print(f"All {len(entries)} tasks submitted. Processing...")
                    
                    # Track completed tasks
                    results = []
                    completed_tasks = 0
                    
                    # Set up progress display
                    with tqdm(total=len(entries), desc="Processing questions") as pbar:
                        remaining_futures = set(futures)
                        
                        while remaining_futures:
                            # Wait for some futures to complete (with timeout)
                            done_futures = set()
                            try:
                                # Use a short timeout to check progress regularly
                                for future in as_completed(remaining_futures, timeout=1.0):
                                    done_futures.add(future)
                                    try:
                                        result = future.result()
                                        if result:
                                            results.append(result)
                                            if output_file:
                                                append_result(result, output_file)
                                            if self.track_cost and "cost" in result:
                                                self.cost_tracker.update_cost(result["cost"])
                                    except Exception as e:
                                        print(f"Error processing entry: {e}")
                                    
                                    # Update progress
                                    completed_tasks += 1
                                    pbar.update(1)
                            except TimeoutError:
                                # No futures completed within timeout - that's okay
                                pass
                            
                            # Remove completed futures
                            remaining_futures -= done_futures
                            
                            # Check if we should stop processing
                            if self.cost_tracker.stop_requested:
                                print(f"\nCost threshold reached. Stopping execution.")
                                for f in remaining_futures:
                                    f.cancel()
                                break
                    
                    return results
            else:
                # Thread-based parallelism (faster startup but less reliable timeouts)
                models = self.create_models(max_workers, use_local_model, use_single_endpoint)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Keep each model bound to one in-flight future at a time.
                    # This avoids submitting the whole dataset up front and avoids
                    # concurrent use of a single API client by multiple threads.
                    def process_func(entry, model_idx):
                        return self.process_entry(
                            entry,
                            models[model_idx],
                            verbose_worker=(model_idx == 0),  # Only first worker shows output
                            **kwargs
                        )

                    next_entry_index = 0
                    in_flight = {}
                    initial_slots = min(max_workers, len(entries))
                    for model_idx in range(initial_slots):
                        future = executor.submit(process_func, entries[next_entry_index], model_idx)
                        in_flight[future] = model_idx
                        next_entry_index += 1

                    with tqdm(total=len(entries), desc="Processing questions") as pbar:
                        while in_flight:
                            done_futures, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                            for future in done_futures:
                                model_idx = in_flight.pop(future)
                                try:
                                    result = future.result()
                                    if result:
                                        results.append(result)
                                        if output_file:
                                            append_result(result, output_file)
                                        if self.track_cost and "cost" in result:
                                            self.cost_tracker.update_cost(result["cost"])
                                except Exception as e:
                                    print(f"Error processing entry: {e}")
                                finally:
                                    pbar.update(1)

                                if self.cost_tracker.stop_requested:
                                    continue
                                if next_entry_index < len(entries):
                                    replacement = executor.submit(
                                        process_func,
                                        entries[next_entry_index],
                                        model_idx,
                                    )
                                    in_flight[replacement] = model_idx
                                    next_entry_index += 1

                            if self.cost_tracker.stop_requested:
                                print("\nCost threshold reached. Stopping execution.")
                                for future in in_flight:
                                    future.cancel()
                                break
        
        return results

    def _process_dataset_isolated(
        self,
        entries: List[Dict],
        *,
        output_file: Optional[str],
        max_workers: int,
        use_local_model: bool,
        use_single_endpoint: bool,
        question_timeout_seconds: float,
        **kwargs,
    ) -> List[Dict]:
        """Process entries in killable, one-question child processes."""
        if not entries:
            return []

        max_workers = max(1, max_workers)
        context = mp.get_context("spawn")
        processor_class_module = self.__class__.__module__
        processor_class_name = self.__class__.__name__
        pending_index = 0
        launched_count = 0
        in_flight = {}
        results = []

        print(
            f"Processing {len(entries)} questions with {max_workers} isolated "
            f"processes (hard timeout={question_timeout_seconds:g}s)"
        )

        def launch(entry):
            nonlocal launched_count
            worker_id = launched_count % max_workers
            launched_count += 1
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=process_entry_in_isolated_process,
                args=(
                    result_queue,
                    entry,
                    worker_id,
                    self.model_kwargs,
                    use_local_model,
                    worker_id == 0,
                    processor_class_module,
                    processor_class_name,
                    use_single_endpoint,
                ),
                kwargs={k: v for k, v in kwargs.items() if k != "self"},
                daemon=False,
            )
            process.start()
            in_flight[process.pid] = {
                "process": process,
                "queue": result_queue,
                "entry": entry,
                "started_at": time.monotonic(),
            }

        def save_result(result):
            if not result:
                return
            results.append(result)
            if output_file:
                append_result(result, output_file)
            if self.track_cost and "cost" in result:
                self.cost_tracker.update_cost(result["cost"])

        def close_job(job, *, terminate=False):
            process = job["process"]
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            job["queue"].close()

        def save_child_error(entry, payload):
            save_result(
                _isolated_failure_result(
                    entry,
                    self.model_kwargs.get("model_id", "unknown"),
                    state="child_process_error",
                    message=(
                        f"{payload.get('error_type', 'Error')}: "
                        f"{payload.get('message', '')}"
                    ),
                    child_traceback=payload.get("traceback"),
                )
            )

        try:
            with tqdm(total=len(entries), desc="Processing questions") as pbar:
                while pending_index < len(entries) or in_flight:
                    while (
                        pending_index < len(entries)
                        and len(in_flight) < max_workers
                        and not self.cost_tracker.stop_requested
                    ):
                        launch(entries[pending_index])
                        pending_index += 1

                    made_progress = False
                    now = time.monotonic()
                    for pid, job in list(in_flight.items()):
                        process = job["process"]
                        result_queue = job["queue"]
                        try:
                            message = result_queue.get_nowait()
                        except Empty:
                            message = None

                        if message is not None:
                            kind, payload = message
                            close_job(job)
                            if kind == "result":
                                save_result(payload)
                            else:
                                save_child_error(job["entry"], payload)
                            del in_flight[pid]
                            pbar.update(1)
                            made_progress = True
                            continue

                        elapsed = now - job["started_at"]
                        if elapsed >= question_timeout_seconds:
                            close_job(job, terminate=True)
                            save_result(
                                _isolated_failure_result(
                                    job["entry"],
                                    self.model_kwargs.get("model_id", "unknown"),
                                    state="execution_timeout",
                                    message=(
                                        "Question exceeded the hard execution timeout "
                                        f"of {question_timeout_seconds:g} seconds"
                                    ),
                                    timeout_seconds=question_timeout_seconds,
                                )
                            )
                            del in_flight[pid]
                            pbar.update(1)
                            made_progress = True
                            continue

                        if not process.is_alive():
                            try:
                                kind, payload = result_queue.get(timeout=0.2)
                            except Empty:
                                kind, payload = "missing", None
                            close_job(job)
                            if kind == "result":
                                save_result(payload)
                            elif kind == "error":
                                save_child_error(job["entry"], payload)
                            else:
                                save_result(
                                    _isolated_failure_result(
                                        job["entry"],
                                        self.model_kwargs.get("model_id", "unknown"),
                                        state="child_process_error",
                                        message=(
                                            "Isolated worker exited without returning a result "
                                            f"(exit code {process.exitcode})"
                                        ),
                                    )
                                )
                            del in_flight[pid]
                            pbar.update(1)
                            made_progress = True

                    if self.cost_tracker.stop_requested:
                        print("\nCost threshold reached. Stopping execution.")
                        break
                    if not made_progress:
                        time.sleep(0.05)
        finally:
            for job in in_flight.values():
                close_job(job, terminate=True)

        return results
