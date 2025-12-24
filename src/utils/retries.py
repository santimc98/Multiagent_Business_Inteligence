import time
from typing import Callable, Any


def call_with_retries(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: int = 1,
    initial_delay: int = 1,
) -> Any:
    """
    Executes a function with exponential backoff retries for specific errors.
    """
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = (
                "429" in error_str
                or "overloaded" in error_str
                or "engine_overloaded_error" in error_str
                or "timeout" in error_str
                or "rate limit" in error_str
                or "concurrency" in error_str
                or "1302" in error_str
            )

            if not is_retryable or attempt >= max_retries:
                raise e

            import random

            delay = initial_delay * (backoff_factor ** attempt)
            jitter = random.uniform(0, delay * 0.25)
            total_delay = delay + jitter

            print(f"API Error: {e}. Retrying in {total_delay:.2f}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(total_delay)
            attempt += 1
