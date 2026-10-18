import copy
import math
import random
import unittest
from unittest.mock import patch

from relation_lab.__main__ import preflight, main

from relation_lab.contracts import (build_relations, canonical_routes, instance_hash,
    relation_bce, shuffled_relation, validate_dataset, validate_routes)


def fixture():
    inst = {"id": "synthetic", "split": "train", "coordinates": [[0, 0], [1, 0], [2, 0], [0, 1]], "demands": [0, 1, 1, 1], "capacity": 2, "distance": "euclidean_float64"}
    cfg = {"M": 2, "duplicates": "keep_equal_weight", "seeds": [11, 12], "budget": {"kind": "iterations", "limit": 1}, "solver": {"name": "synthetic-fixture", "version": "1", "binary_sha256": "0" * 64}}
    refs = [{"instance_id": inst["id"], "instance_hash": instance_hash(inst), "status": "success", "source": "in-memory fixture, no solver executed", "solver": cfg["solver"], "budget": cfg["budget"], "seed": seed, "routes": routes, "cost": validate_routes(inst, routes)} for seed, routes in zip(cfg["seeds"], [[[1, 2], [3]], [[1], [2, 3]]])]
    return inst, cfg, refs


class Contracts(unittest.TestCase):
    def test_cost_and_feasibility(self):
        inst, _, _ = fixture()
        self.assertEqual(validate_routes(inst, [[1, 2], [3]]), 6)
        for bad in ([[1, 2, 3]], [[1, 1], [2, 3]], [[1, 2]], [[0, 1], [2, 3]], [[True, 2], [3]], [[], [1, 2], [3]]):
            with self.assertRaises(ValueError):
                validate_routes(inst, bad)

    def test_relations_and_invariance(self):
        inst, cfg, refs = fixture()
        expected = [[0, .5, 0], [.5, 0, .5], [0, .5, 0]]
        self.assertEqual(build_relations(inst, refs, cfg)["p"], expected)
        modified = copy.deepcopy(refs)
        for ref in modified:
            ref["routes"] = [list(reversed(r)) for r in reversed(ref["routes"])]
        self.assertEqual(build_relations(inst, modified, cfg)["p"], expected)
        self.assertEqual(canonical_routes(refs[0]["routes"]), canonical_routes(modified[0]["routes"]))

    def test_fail_closed(self):
        inst, cfg, refs = fixture()
        for field, value in (("instance_hash", "wrong"), ("instance_id", "wrong"), ("status", "failed"), ("seed", 99), ("cost", 9), ("source", ""), ("budget", {"kind": "seconds", "limit": 1})):
            bad = copy.deepcopy(refs)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                build_relations(inst, bad, cfg)
        with self.assertRaises(ValueError):
            build_relations(inst, refs[:1], cfg)
        inst["split"] = "test"
        with self.assertRaises(ValueError):
            build_relations(inst, refs, cfg)

    def test_duplicates(self):
        inst, cfg, refs = fixture()
        refs[1]["routes"] = [[3], [2, 1]]
        refs[1]["cost"] = refs[0]["cost"]
        self.assertEqual(build_relations(inst, refs, cfg)["duplicates"], 1)
        cfg["duplicates"] = "reject"
        with self.assertRaises(ValueError):
            build_relations(inst, refs, cfg)

    def test_split_hash(self):
        inst, _, _ = fixture()
        other = copy.deepcopy(inst)
        other.update(id="renamed", split="test")
        self.assertEqual(instance_hash(inst), instance_hash(other))
        other["capacity"] = 2.0
        self.assertEqual(instance_hash(inst), instance_hash(other))
        with self.assertRaises(ValueError):
            validate_dataset([inst, other])
        other["coordinates"][1][0] = 3
        validate_dataset([inst, other])

    def test_shuffle_private_rng(self):
        inst, cfg, refs = fixture()
        p = build_relations(inst, refs, cfg)["p"]
        state = random.getstate()
        shuffled, perm = shuffled_relation(p, instance_hash(inst), 43)
        self.assertEqual(random.getstate(), state)
        self.assertEqual((shuffled, perm), shuffled_relation(p, instance_hash(inst), 43))
        self.assertEqual(sorted(x for r in p for x in r), sorted(x for r in shuffled for x in r))
        for i in range(3):
            for j in range(3):
                self.assertEqual(shuffled[i][j], p[perm[i]][perm[j]])

    def test_loss(self):
        inst, cfg, refs = fixture()
        p = build_relations(inst, refs, cfg)["p"]
        self.assertAlmostEqual(relation_bce([[0]*3 for _ in range(3)], p), math.log(2))
        self.assertTrue(math.isfinite(relation_bce([[1000]*3 for _ in range(3)], p)))
        with self.assertRaises(ValueError):
            relation_bce([[0, 1, 0], [0, 0, 0], [0, 0, 0]], p)

    def test_preflight_without_disk(self):
        inst, cfg, refs = fixture()
        with patch("relation_lab.__main__.read", side_effect=[{"schema": 1, "reference": cfg}, [inst], {inst["id"]: refs}]):
            result = preflight("config", "data", "references")
        self.assertIn(inst["id"], result[3])
        with patch("relation_lab.__main__.read", side_effect=[{"schema": 1, "reference": cfg}, [inst], {"test-reference": refs}]):
            with self.assertRaises(ValueError):
                preflight("config", "data", "references")

    def test_real_entrypoints_blocked(self):
        for command in ("generate-references", "train", "evaluate"):
            with patch("sys.argv", ["relation_lab", command, "--confirm-real-run"]), patch("sys.stderr"), patch("relation_lab.__main__.read") as reader:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
