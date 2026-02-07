#!/usr/bin/env python3
"""
Backward-compatibility shim for legacy imports.

Runtime code should import from `src.utils.contract_accessors`.
This module re-exports the same symbols to keep old call sites/tests working
during the migration window.
"""

from src.utils.contract_accessors import *  # noqa: F401,F403

