import os
import threading
import time
import tempfile
from contextlib import contextmanager

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


_glm_max = int(os.getenv("GLM_MAX_CONCURRENCY", "1") or "1")
_glm_max = max(1, _glm_max)
_glm_semaphore = threading.Semaphore(_glm_max)
_glm_use_file_lock = os.getenv("GLM_USE_FILE_LOCK", "1").strip().lower() not in {"0", "false", "no"}
_glm_lock_timeout = float(os.getenv("GLM_LOCK_TIMEOUT", "120") or "120")
_glm_lock_sleep = float(os.getenv("GLM_LOCK_SLEEP", "0.25") or "0.25")
_glm_lock_path = os.getenv("GLM_LOCK_PATH") or os.path.join(tempfile.gettempdir(), "glm_api.lock")


def _acquire_file_lock(lock_file):
    if not _glm_use_file_lock:
        return
    if msvcrt is None and fcntl is None:
        return
    deadline = time.monotonic() + _glm_lock_timeout
    while True:
        try:
            if msvcrt is not None:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            elif fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for GLM API lock.")
            time.sleep(_glm_lock_sleep)


def _release_file_lock(lock_file):
    if not _glm_use_file_lock:
        return
    if msvcrt is None and fcntl is None:
        return
    try:
        if msvcrt is not None:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _open_lock_file():
    os.makedirs(os.path.dirname(_glm_lock_path), exist_ok=True)
    lock_file = open(_glm_lock_path, "a+b")
    if lock_file.tell() == 0:
        lock_file.write(b"0")
        lock_file.flush()
    return lock_file


@contextmanager
def glm_call_slot():
    _glm_semaphore.acquire()
    lock_file = None
    try:
        lock_file = _open_lock_file()
        _acquire_file_lock(lock_file)
        yield
    finally:
        if lock_file is not None:
            _release_file_lock(lock_file)
            lock_file.close()
        _glm_semaphore.release()
