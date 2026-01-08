"""
Test script to validate senior reasoning in Strategist Agent.

This tests the new first-principles reasoning approach vs old keyword matching.
"""

import sys
sys.path.insert(0, 'src')

from agents.strategist import StrategistAgent
import json

# Test case from run_ec8b74fe (previously failed)
data_summary = """
DATA SUMMARY:
The dataset consists of 352 records and 246 columns representing commercial interactions for the "Trinium" product.
Key variables: CurrentPhase (outcome), 1stYearAmount (offered price), Size (annual turnover), Debtors (number of debtors), Sector (industry).
Probability field is an operational score with potential leakage risk.
"""

user_request = """
El objetivo es construir un modelo de precio que relacione el valor ofertado ("1stYearAmount")
con la tipología del cliente —principalmente "Size", "Debtors" y "Sector"— para estimar el
precio óptimo: el mayor precio al que cada tipología compraría el producto con la mayor
probabilidad de éxito posible.
"""

print("=" * 80)
print("TESTING STRATEGIST SENIOR REASONING")
print("=" * 80)

print("\n[TEST CASE]")
print(f"User Request: {user_request[:200]}...")
print(f"\nData Summary: {data_summary[:200]}...")

try:
    strategist = StrategistAgent()
    print("\n[GENERATING STRATEGY]")
    result = strategist.generate_strategies(data_summary, user_request)

    print("\n[RESULT]")
    strategies = result.get("strategies", [])
    if strategies:
        strategy = strategies[0]

        print(f"\nTitle: {strategy.get('title')}")
        print(f"Objective Type: {strategy.get('objective_type')}")
        print(f"\nObjective Reasoning:")
        print(f"  {strategy.get('objective_reasoning', 'N/A')}")
        print(f"\nSuccess Metric: {strategy.get('success_metric')}")
        print(f"Recommended Metrics: {strategy.get('recommended_evaluation_metrics')}")
        print(f"Validation Strategy: {strategy.get('validation_strategy')}")

        # Check strategy_spec
        strategy_spec = result.get("strategy_spec", {})
        print(f"\n[STRATEGY_SPEC]")
        print(f"Objective Type: {strategy_spec.get('objective_type')}")
        print(f"Metrics: {strategy_spec.get('evaluation_plan', {}).get('metrics')}")

        # Validate
        print("\n[VALIDATION]")
        expected_objective = "prescriptive"
        actual_objective = strategy.get('objective_type', '').lower()

        if expected_objective in actual_objective:
            print(f"PASS: Objective type is '{actual_objective}' (expected prescriptive/optimization)")
        else:
            print(f"FAIL: Objective type is '{actual_objective}' (expected prescriptive, not forecasting/regression)")

        if strategy.get('objective_reasoning'):
            print(f"PASS: Objective reasoning provided")
            print(f"   Reasoning: {strategy['objective_reasoning'][:150]}...")
        else:
            print(f"FAIL: No objective reasoning provided")

        if "Expected" in str(strategy.get('success_metric', '')) or "revenue" in str(strategy.get('success_metric', '')).lower():
            print(f"PASS: Success metric mentions Expected Value or Revenue")
        else:
            print(f"WARNING: Success metric might not be optimal for optimization: {strategy.get('success_metric')}")

    else:
        print("❌ FAIL: No strategies generated")

    # Save full output for inspection
    with open("test_strategist_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n[FULL OUTPUT SAVED TO: test_strategist_output.json]")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
