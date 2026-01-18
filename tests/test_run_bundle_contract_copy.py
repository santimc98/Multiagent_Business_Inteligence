import json
import os

from src.utils.run_bundle import copy_run_contracts, init_run_bundle


def test_copy_run_contracts_uses_absolute_sources(tmp_path):
    run_id = "test_run_abs_contracts"
    run_dir = tmp_path / "runs" / run_id
    init_run_bundle(run_id, run_dir=str(run_dir), enable_tee=False)

    work_dir = tmp_path / "work"
    work_data = work_dir / "data"
    work_data.mkdir(parents=True, exist_ok=True)
    contract_a = {"strategy_title": "Strategy A"}
    with open(work_data / "execution_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_a, f)

    global_data = tmp_path / "data"
    global_data.mkdir(parents=True, exist_ok=True)
    contract_b = {"strategy_title": "Strategy B"}
    with open(global_data / "execution_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_b, f)

    copy_run_contracts(run_id, [os.path.abspath(work_data / "execution_contract.json")])

    copied_path = run_dir / "contracts" / "execution_contract.json"
    assert copied_path.exists()
    with open(copied_path, "r", encoding="utf-8") as f:
        copied = json.load(f)
    assert copied.get("strategy_title") == "Strategy A"
