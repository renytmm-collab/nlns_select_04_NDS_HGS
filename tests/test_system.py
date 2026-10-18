import ast
import copy
import itertools
import json
from pathlib import Path
import random
import unittest
import uuid
from unittest.mock import patch

from relation_lab.contracts import digest, instance_hash, validate_routes
from relation_lab.evaluation import paired_report
from relation_lab.experiment import main
from relation_lab.fixtures import smoke_bundle
from relation_lab.references import generate, solver_identity
from relation_lab.runtime import gate, labels_for, read_json, validate_config, validate_inputs
from relation_lab.search import search
from relation_lab.vrp import action_mask, distances, insert_sequence, leave_one_out, ordinary_state, repair, route_cost


class Mechanics(unittest.TestCase):
    def setUp(self):
        self.c, self.data, self.states, self.refs = smoke_bundle()
        self.i, self.routes = self.data[0], self.states[0]["routes"]

    def test_compile_all_sources_without_cache(self):
        for folder in ("relation_lab", "tests"):
            for path in Path(folder).glob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_mask_and_repair_properties(self):
        rng = random.Random(42)
        for k in range(1, 9):
            for _ in range(5):
                action = rng.sample(range(1, 9), k)
                mask = action_mask(8, action)
                self.assertEqual(sum(mask), k+1)
                before = copy.deepcopy(self.routes)
                result = repair(self.i, self.routes, action)
                self.assertEqual(before, self.routes)
                validate_routes(self.i, result)
        for action in ([0], [1, 1], [9]):
            with self.assertRaises(ValueError):
                repair(self.i, self.routes, action)

    def test_order_changes_cost(self):
        costs = {round(validate_routes(self.i, repair(self.i, self.routes, list(order))), 10) for order in itertools.permutations([1, 2, 4, 7])}
        self.assertGreater(len(costs), 1, "same deleted set can have distinct reinsertion outcomes")

    def test_insertion_ties_and_capacity(self):
        inst = {"id": "tie", "split": "train", "coordinates": [[0, 0]]*4, "demands": [0, 1, 1, 1], "capacity": 2, "distance": "euclidean_float64"}
        self.assertEqual(insert_sequence(inst, [[1], [2]], [3]), [[3, 1], [2]])
        self.assertEqual(insert_sequence(inst, [[1, 2]], [3]), [[1, 2], [3]])
        self.assertEqual(repair(inst, [[1, 2], [3]], [1, 2, 3]), [[2, 1], [3]])

    def test_whitelist_and_family(self):
        poisoned = dict(self.i, p_ij=[[999]], reference_path="missing-file", reference_cost=1)
        self.assertEqual(ordinary_state(self.i, self.routes), ordinary_state(poisoned, self.routes))
        data = copy.deepcopy(self.data)
        data[-1]["family_id"] = data[0]["family_id"]
        with self.assertRaises(ValueError):
            validate_inputs(self.c, data, self.states)

    def test_loo_unclipped_and_normalization(self):
        self.assertEqual(leave_one_out([1, -1, 0]), [1.5, -1.5, 0])
        with self.assertRaises(ValueError):
            leave_one_out([1])

    def test_config_gates(self):
        validate_inputs(self.c, self.data, self.states)
        gate(self.c, True, False)
        with self.assertRaises(ValueError):
            gate(self.c, False, True)
        config = copy.deepcopy(self.c)
        config["training"]["steps"] = 999
        with self.assertRaises(ValueError):
            validate_config(config)
        config = copy.deepcopy(self.c)
        config.update(kind="experiment", budgets_confirmed=True)
        gate(config, False, True)
        with self.assertRaises(ValueError):
            gate(config, False, False)

    def test_search_monotone_and_counts(self):
        calls = []
        def choose(routes):
            calls.append(routes)
            return [1, 2, 4]
        result = search(self.i, self.routes, choose, mode="iterations", iterations=3)
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(len(calls), 3)
        costs = [p["cost"] for p in result["trace"]]
        self.assertEqual(costs, sorted(costs, reverse=True))
        validate_routes(self.i, result["routes"])

    def test_late_improvement_not_credited(self):
        ticks = iter([0, 0, 2, 2])
        result = search(self.i, self.routes, lambda _: [1, 2, 4], mode="time", seconds=1, clock=lambda: next(ticks))
        self.assertTrue(result["late_candidate_discarded"])
        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["routes"], self.routes)
        self.assertEqual(result["overshoot_seconds"], 1)

    def test_generator_deterministic_budget_no_policy_rng(self):
        cfg = copy.deepcopy(self.c["reference"])
        cfg["solver"] = solver_identity()
        before = random.getstate()
        a, b = generate(self.i, cfg, 2), generate(self.i, cfg, 2)
        self.assertEqual(before, random.getstate())
        self.assertEqual([r["routes"] for r in a], [r["routes"] for r in b])
        self.assertTrue(all(r["starts_completed"] == 1 for r in a))
        bad = dict(self.i, split="test")
        with self.assertRaises(ValueError):
            generate(bad, cfg, 2)

    def test_paired_stats_cluster_instances(self):
        rows = [{"instance_hash": h, "state_hash": h, "kind": "single", "eval_seed": 2, "repeat": r, "initial_cost": 10, "cost": cost} for h, cost in [("a", 5), ("b", 7)] for r in range(3)]
        left = [{"train_seed": 1, "protocol_hash": "same", "rows": rows}]
        right = copy.deepcopy(left)
        for r in right[0]["rows"]:
            r["cost"] += 2
        result = paired_report(left, right, bootstrap_samples=100)
        self.assertEqual(result["single"]["instances"], 2)
        self.assertEqual(result["single"]["paired_mean_cost_difference"], 2)
        right[0]["rows"].pop()
        with self.assertRaises(ValueError):
            paired_report(left, right)


class DiskContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("test_artifacts") / ("contracts-" + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.bundle = cls.root / "fixture"
        with patch("sys.stdout"):
            assert main(["make-smoke", "--run-dir", str(cls.bundle)]) == 0

    def base(self, group="baseline"):
        return ["--config", str(self.bundle/(group+".json")), "--data", str(self.bundle/"data.json"), "--states", str(self.bundle/"states.json")]

    def test_preflight_readonly_and_reference_boundary(self):
        before = set(self.root.rglob("*"))
        with patch("sys.stdout"):
            self.assertEqual(main(["preflight", *self.base()]), 0)
            self.assertEqual(main(["preflight", *self.base("relation"), "--references", str(self.bundle/"references.json")]), 0)
        self.assertEqual(before, set(self.root.rglob("*")))
        with patch("sys.stderr"):
            self.assertEqual(main(["preflight", "--task", "evaluate", *self.base(), "--references", "file-must-not-be-opened.json"]), 1)

    def test_existing_directory_and_no_start_without_gate(self):
        with patch("sys.stderr"):
            self.assertEqual(main(["make-smoke", "--run-dir", str(self.bundle)]), 1)
            self.assertEqual(main(["train", *self.base(), "--run-dir", str(self.root/"must-not-exist")]), 1)
        self.assertFalse((self.root/"must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
