"""Bounded pool of isolated Rasa processes using a line-delimited JSON protocol."""

import atexit
import json
import os
import queue
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

POOL_SIZE = 2
_pool = OrderedDict()
_pool_lock = threading.Lock()


class Worker:
    def __init__(self, python: str, script: Path, model: str):
        self.lock = threading.Lock()
        self.output = queue.Queue()
        self.process = subprocess.Popen(
            [python, str(script), model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env={
                **os.environ,
                "RASA_TELEMETRY_ENABLED": "false",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
            },
        )
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        if self.process.stdout:
            for line in self.process.stdout:
                if line.startswith("ATLAS_RESULT "):
                    self.output.put(line.removeprefix("ATLAS_RESULT "))
        self.output.put(None)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def parse(self, text: str):
        if not self.process.stdin:
            raise RuntimeError("Канал Rasa недоступен")
        self.process.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.output.get(timeout=120)
        except queue.Empty as exc:
            raise RuntimeError("Превышено время ожидания Rasa") from exc
        if line is None:
            raise RuntimeError("Процесс Rasa завершился. Проверьте окружение и модель.")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result


def parse(python: str, script: Path, model: str, text: str):
    # Reserve a worker while holding the pool lock so it cannot be evicted in use.
    with _pool_lock:
        worker = _pool.pop(model, None)
        if worker is not None and worker.process.poll() is not None:
            worker.close()
            worker = None
        if worker is None:
            if len(_pool) >= POOL_SIZE:
                _, old = _pool.popitem(last=False)
                with old.lock:
                    old.close()
            worker = Worker(python, script, model)
        _pool[model] = worker
        worker.lock.acquire()
    try:
        return worker.parse(text)
    except OSError, RuntimeError, ValueError:
        worker.close()
        raise
    finally:
        worker.lock.release()


def close_all():
    with _pool_lock:
        for worker in _pool.values():
            with worker.lock:
                worker.close()
        _pool.clear()


atexit.register(close_all)
