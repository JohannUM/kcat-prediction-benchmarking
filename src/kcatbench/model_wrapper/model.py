import logging
import os
import pandas as pd
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from kcatbench.model_wrapper.base_model import BaseModel
from kcatbench.model_wrapper.subprocess_logging import (
    WorkerLogEvent,
    parse_worker_log_line,
    render_worker_log_event,
)
from kcatbench.util import read_csv_with_schema

ENVIRONMENT_NAMES: dict[str, str] = {
    "dlkcat": "dlkcat_env",
    "catapro": "catapro_env",
    "catpred": "catpred_env",
    "turnup": "turnup_env",
    "unikp": "unikp_env"
}

class Model(BaseModel):

    @dataclass(frozen=True)
    class SubprocessLogPolicy:
        progress_event_prefix: str = "progress."

        def should_forward(self, event: WorkerLogEvent) -> bool:
            if not event.live:
                return False
            if not event.event:
                return False
            return event.event.startswith(self.progress_event_prefix)

    @dataclass
    class _SubprocessCapture:
        returncode: int
        stdout_lines: list[str]
        stderr_lines: list[str]

    def __init__(
        self,
        model_name: str,
        env_name: str = None,
        logger: Optional[logging.Logger] = None,
        log_policy: Optional["Model.SubprocessLogPolicy"] = None,
    ):
        if(model_name not in ENVIRONMENT_NAMES.keys()):
            raise ValueError(f"{model_name} is not valid\nValid names are: {ENVIRONMENT_NAMES.keys()}")

        if(env_name != None):
            self.env_name = env_name
        else:
            self.env_name = ENVIRONMENT_NAMES[model_name]
        self.model_name = model_name
        self.logger = logger or logging.getLogger("kcatbench.model_wrapper")
        self.log_policy = log_policy or Model.SubprocessLogPolicy()
        super().__init__()

    def _handle_stderr_line(self, line: str, stderr_lines: list[str]) -> None:
        event = parse_worker_log_line(line)
        if event is None:
            stderr_lines.append(line)
            return

        rendered = render_worker_log_event(event)
        stderr_lines.append(rendered)

        if self.log_policy.should_forward(event):
            self.logger.log(event.levelno, "[%s] %s", self.model_name, rendered)

    def _stream_worker_subprocess(self, cmd: list[str]) -> "Model._SubprocessCapture":
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._get_isolated_env(),
            bufsize=1,
        )

        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Failed to capture worker subprocess output streams.")

        line_queue: Queue[tuple[str, str]] = Queue()

        def _enqueue_stream(stream_name: str, stream) -> None:
            try:
                for raw_line in iter(stream.readline, ""):
                    line_queue.put((stream_name, raw_line.rstrip("\n")))
            finally:
                stream.close()

        stdout_thread = threading.Thread(
            target=_enqueue_stream,
            args=("stdout", process.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_enqueue_stream,
            args=("stderr", process.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        while True:
            try:
                stream_name, line = line_queue.get(timeout=0.1)
                if stream_name == "stdout":
                    stdout_lines.append(line)
                else:
                    self._handle_stderr_line(line, stderr_lines)
            except Empty:
                pass

            process_finished = process.poll() is not None
            streams_drained = not stdout_thread.is_alive() and not stderr_thread.is_alive() and line_queue.empty()
            if process_finished and streams_drained:
                break

        stdout_thread.join()
        stderr_thread.join()

        while not line_queue.empty():
            stream_name, line = line_queue.get_nowait()
            if stream_name == "stdout":
                stdout_lines.append(line)
            else:
                self._handle_stderr_line(line, stderr_lines)

        return Model._SubprocessCapture(
            returncode=process.wait(),
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
        )

    def _log_subprocess_dump(self, channel: str, lines: list[str]) -> None:
        if not lines:
            return
        self.logger.error(
            "[%s] Worker %s dump follows:\n%s",
            self.model_name,
            channel,
            "\n".join(lines),
        )

    def _get_isolated_env(self) -> dict[str, str]:
        """Creates a clean environment dictionary for the subprocess."""
        env = os.environ.copy()
        
        keys_to_remove = [
            "PYTHONPATH",        
            "PYTHONHOME", 
            "MPLBACKEND",
        ]
        
        for key in keys_to_remove:
            env.pop(key, None)
            
        return env

    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.csv"
            output_path = tmpdir / "output.csv"

            input_data.to_csv(input_path, index=False)

            cmd = [
                "conda", "run", "-n", self.env_name, "--no-capture-output",
                "python", "-m", "kcatbench.model_wrapper.model_worker", "predict",
                "--model", self.model_name, "--input", str(input_path), "--output", str(output_path)
            ]

            capture = self._stream_worker_subprocess(cmd)

            if(capture.returncode != 0):
                self._log_subprocess_dump("stdout", capture.stdout_lines)
                self._log_subprocess_dump("stderr", capture.stderr_lines)

                raise RuntimeError(
                    f"Model {self.model_name} failed with exit code {capture.returncode}:"
                    f"\nOUTPUT:\n{'\n'.join(capture.stdout_lines)}"
                    f"\n\nERROR:\n{'\n'.join(capture.stderr_lines)}"
                )

            return read_csv_with_schema(output_path)