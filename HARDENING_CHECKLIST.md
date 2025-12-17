# System Hardening Checklist

Run these commands to verify the integrity of the Multi-Agent System.

## 1. Deduplication & Routing
Verify that `graph.py` and other core agents are unique and safe.

```bash
# Check for "subprocess" usage in src
grep -r "subprocess" src/
# Expected: src/graph/graph.py (only inside static_safety_scan list blocking it) or 0 results if fully clean.

# Check that run_domain_expert exists in graph
grep "def run_domain_expert" src/graph/graph.py
# Expected: Match found

# Check that static_safety_scan exists in graph
grep "def static_safety_scan" src/graph/graph.py
# Expected: Match found
```

## 2. Robust CSV Handling
Verify agents accept CSV metadata (Separator, Decimal).

```bash
# Check Data Engineer signature
grep "csv_sep" src/agents/data_engineer.py
# Expected: Match found in verify_cleaning_script signature

# Check ML Engineer signature
grep "csv_sep" src/agents/ml_engineer.py
# Expected: Match found in generate_code signature
```

## 3. Fail-Safe Mechanisms
Verify that review loops are bounded.

```bash
# Check Reviewer Fail-Safe in graph.py
grep -C 5 "if current_iter + 1 > 3" src/graph/graph.py
# Expected: Logic returning error_message
```

## 4. Sandbox Security
Verify E2B usage for cleaning.

```bash
# Check Sandbox creation in Run Data Engineer
grep "Sandbox.create()" src/graph/graph.py
# Expected: Found in run_data_engineer and execute_code
```

## 5. Domain Expert Deliberation
Verify Domain Expert Agent implementation.

```bash
# Check evaluate_strategies method
grep "def evaluate_strategies" src/agents/domain_expert.py
# Expected: Match found
```
