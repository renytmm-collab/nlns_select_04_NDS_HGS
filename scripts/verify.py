"""Run synthetic tests; strict server mode forbids missing neural tests."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-neural", action="store_true")
    p.add_argument("--report-dir")
    args = p.parse_args()
    if args.require_neural and importlib.util.find_spec("torch") is None:
        print("FAILED: PyTorch unavailable; strict neural validation not performed", file=sys.stderr)
        return 2
    if Path.cwd().resolve() != ROOT:
        print("Run this command from PROJECT_ROOT", file=sys.stderr)
        return 2
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/"tests")))
    record = {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
              "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
              "seconds": time.perf_counter()-start, "synthetic_only": True,
              "strict_neural": args.require_neural, "python": sys.version}
    if args.report_dir:
        from relation_lab.runtime import Run, code_identity, write_json
        from relation_lab.contracts import digest
        run = Run(args.report_dir, "synthetic_verification", {"code": digest(code_identity())}, {})
        write_json(run.path/"verification.json", record)
        run.finish("complete" if result.wasSuccessful() else "failed")
    print(json.dumps(record, indent=2))
    return 0 if result.wasSuccessful() and not (args.require_neural and result.skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
