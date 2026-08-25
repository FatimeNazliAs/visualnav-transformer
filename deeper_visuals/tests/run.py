# deeper_visuals/tests/run.py
"""
Run the library's tests.

    python3 -m deeper_visuals.tests.run

Deliberately not pytest: the container's `vint_train` environment does not have
it, and adding a dependency to run four files of asserts would put the tests
behind an environment change — which is the sort of friction that ends with
nobody running them.

Nothing here loads torch, a checkpoint, the dataset or a GPU. That is the point:
`common/smoke_test.py` proves the model still works and needs the whole rig;
this proves the arithmetic and the page contract still hold, and needs none of
it. Both are worth having, and only one can run in a second.
"""

from __future__ import annotations

import sys
import traceback

from deeper_visuals.tests import test_actions, test_copy_contract

MODULES = (test_actions, test_copy_contract)


def main() -> int:
    passed, failures = 0, []

    for module in MODULES:
        names = sorted(n for n in dir(module) if n.startswith("test_"))
        print(f"\n  {module.__name__}  ({len(names)} tests)")
        for name in names:
            try:
                getattr(module, name)()
            except Exception:
                failures.append((module.__name__, name, traceback.format_exc()))
                print(f"    FAIL  {name}")
            else:
                passed += 1
                print(f"    ok    {name}")

    print()
    for module_name, name, tb in failures:
        print(f"─── {module_name}.{name} " + "─" * 40)
        print(tb)

    total = passed + len(failures)
    print(f"  {passed}/{total} passed"
          + (f", {len(failures)} FAILED" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
