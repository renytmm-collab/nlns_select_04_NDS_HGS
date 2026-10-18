import contextlib
import importlib.util
import io
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

from relation_lab.runtime import labels_for, read_json, validate_inputs

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_xml100.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_xml100", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_zip(path, count=12, customers=8):
    module = load_script()
    from relation_lab.contracts import validate_routes
    from relation_lab.vrp import insert_sequence
    with zipfile.ZipFile(path, "w") as archive:
        for j in range(count):
            rng = random.Random(500 + j)
            xy = [[rng.randint(0, 1000), rng.randint(0, 1000)] for _ in range(customers + 1)]
            demand = [0] + [rng.randint(1, 3) for _ in range(customers)]
            stem = "XML100_1111_%02d" % (j + 1)
            lines = ["NAME : %s" % stem, "COMMENT : synthetic", "TYPE : CVRP", "DIMENSION : %d" % (customers + 1),
                     "EDGE_WEIGHT_TYPE : EUC_2D", "CAPACITY : 6", "NODE_COORD_SECTION"]
            lines += ["%d %d %d" % (i + 1, x, y) for i, (x, y) in enumerate(xy)]
            lines += ["DEMAND_SECTION"] + ["%d %d" % (i + 1, d) for i, d in enumerate(demand)]
            lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
            archive.writestr("Vrp-Set-XML100/instances/%s.vrp" % stem, "\n".join(lines) + "\n")
            inst = {"coordinates": [[float(a), float(b)] for a, b in xy], "demands": [float(d) for d in demand],
                    "capacity": 6.0, "distance": "euclidean_float64", "split": "train", "id": stem}
            routes = insert_sequence(inst, [], list(range(1, customers + 1)))
            validate_routes(inst, routes)
            body = ["Route #%d: %s" % (k + 1, " ".join(map(str, r))) for k, r in enumerate(routes)]
            body.append("Cost %d" % module.rounded_cost(inst, routes))
            archive.writestr("Vrp-Set-XML100/solutions/%s.sol" % stem, "\n".join(body) + "\n")


class PrepareXml100(unittest.TestCase):
    def test_end_to_end_contracts(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            archive, out = Path(tmp) / "x.zip", Path(tmp) / "out"
            synthetic_zip(archive)
            with contextlib.redirect_stdout(io.StringIO()):
                code = module.main(["--out-dir", str(out), "--zip", str(archive), "--train", "6", "--validation", "2",
                                    "--test", "4", "--customers", "8", "--destroy-size", "3", "--train-extra-states", "2"])
            self.assertEqual(code, 0)
            data, states, refs = (read_json(out / (n + ".json")) for n in ("data", "states", "references"))
            splits = {s: {d["id"] for d in data if d["split"] == s} for s in ("train", "validation", "test")}
            self.assertEqual([len(v) for v in splits.values()], [6, 2, 4])
            self.assertFalse(splits["train"] & splits["test"] or splits["train"] & splits["validation"] or splits["validation"] & splits["test"])
            self.assertEqual(set(refs), splits["train"])
            for group, weight in (("baseline", 0.0), ("relation", 0.1), ("shuffled_relation", 0.1)):
                config = read_json(out / (group + ".json"))
                self.assertEqual((config["group"], config["lambda"], config["budgets_confirmed"]), (group, weight, False))
                validate_inputs(config, data, states)
            labels = labels_for(read_json(out / "relation.json"), data, refs)
            self.assertEqual(set(labels), splits["train"])
            held_out = {s["instance_id"] for s in states if s["instance_id"] not in splits["train"]}
            self.assertEqual(sum(s["instance_id"] in held_out for s in states), 6)  # one state per val/test instance

    def test_wrong_customer_count_rejected(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            archive, out = Path(tmp) / "x.zip", Path(tmp) / "out"
            synthetic_zip(archive)
            with self.assertRaises(SystemExit):
                module.main(["--out-dir", str(out), "--zip", str(archive), "--train", "6", "--validation", "2", "--test", "4"])


if __name__ == "__main__":
    unittest.main()
