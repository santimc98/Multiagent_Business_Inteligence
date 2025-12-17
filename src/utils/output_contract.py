import glob
import os
from typing import List, Dict


def check_required_outputs(required_outputs: List[str]) -> Dict[str, object]:
    """
    Best-effort validation of required outputs.
    Supports glob patterns (e.g., static/plots/*.png).
    Returns dict with present, missing, summary.
    Never raises.
    """
    present: List[str] = []
    missing: List[str] = []
    required_outputs = required_outputs or []

    for pattern in required_outputs:
        try:
            if any(char in pattern for char in ["*", "?", "["]):
                matches = glob.glob(pattern)
                if matches:
                    present.extend(matches)
                else:
                    missing.append(pattern)
            else:
                if os.path.exists(pattern):
                    present.append(pattern)
                else:
                    missing.append(pattern)
        except Exception:
            missing.append(pattern)

    summary = f"Present: {len(present)}; Missing: {len(missing)}"
    return {"present": present, "missing": missing, "summary": summary}
