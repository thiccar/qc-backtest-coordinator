"""Code that can be shared among Jupyter notebooks"""
from coordinator import CoordinatorIO
from testsets import Test, TestState


def load_results(test_dir, keep_useful: True):
    c = CoordinatorIO(test_dir)

    state = c.read_state()
    tests = [Test.from_dict(t) for t in state["coordinator"]["tests"]]
    results = []

    for t in tests:
        if t.state == TestState.RUNNING: # Allows us to run this while backtests are running to see intermediate results
            continue
        result = c.read_test_result(t)

        # If the definition of useful varies a lot across tests, we may need to be more flexible here
        if keep_useful:
            useful = {k: result.bt_result[k] for k in result.required_keys}
            useful["charts"] = {"Strategy Equity": result.bt_result["charts"]["Strategy Equity"]}
            result.bt_result = useful
        results.append(result)

    return results
