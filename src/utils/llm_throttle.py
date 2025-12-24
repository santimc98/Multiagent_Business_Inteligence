import os
import threading
from contextlib import contextmanager


_glm_max = int(os.getenv("GLM_MAX_CONCURRENCY", "1") or "1")
_glm_max = max(1, _glm_max)
_glm_semaphore = threading.Semaphore(_glm_max)


@contextmanager
def glm_call_slot():
    _glm_semaphore.acquire()
    try:
        yield
    finally:
        _glm_semaphore.release()
