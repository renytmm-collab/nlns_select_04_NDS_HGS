"""Read-only validation and exclusive run artifacts; no torch import."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from .contracts import build_relations, digest, instance_hash, require, validate_dataset, validate_routes


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def positive(value, name, integer=False, zero=False):
    require(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and (value >= 0 if zero else value > 0), "invalid " + name)


def validate_config(c):
    require(c["schema"] == 2, "experiment schema must be 2")
    require(c["kind"] in ("synthetic", "experiment"), "invalid run kind")
    require(c["distance"] == "euclidean_float64", "distance must be euclidean_float64")
    require(isinstance(c["distribution"], str) and c["distribution"].strip(), "explicit distribution required")
    positive(c["customers"], "customers", integer=True)
    require(c["customers"] >= 2, "relation task needs at least two customers")
    positive(c["destroy_size"], "destroy_size", integer=True)
    require(c["destroy_size"] <= c["customers"], "destroy size exceeds N")
    require(c["group"] in ("baseline", "relation", "shuffled_relation"), "unknown group")
    positive(c["lambda"], "lambda", zero=True)
    require(c["group"] != "baseline" or c["lambda"] == 0, "baseline lambda must be zero")
    for key in ("seed", "shuffle_seed"):
        positive(c[key], key, integer=True, zero=True)
        require(c[key] < 2**31, "seed out of range")
    m, t, e = c["model"], c["training"], c["evaluation"]
    require(set(m) == {"width", "heads", "layers"}, "model keys")
    for key in m:
        positive(m[key], key, integer=True)
    require(m["width"] % m["heads"] == 0, "width must divide heads")
    for key in ("steps", "batch_size", "rollouts", "threads"):
        positive(t[key], key, integer=True)
    require(t["rollouts"] >= 2, "LOO requires >=2 rollouts")
    for key in ("learning_rate", "grad_clip"):
        positive(t[key], key)
    require(t["device"] in ("cpu", "cuda"), "device cpu or cuda required")
    require(t["checkpoint_rule"] == "last_step", "only prespecified last_step selection supported")
    for key in ("repeats", "iterations"):
        positive(e[key], key, integer=True)
    positive(e["seconds"], "seconds")
    require(isinstance(e["seeds"], list) and e["seeds"] and len(e["seeds"]) == len(set(e["seeds"])), "unique eval seeds required")
    for seed in e["seeds"]:
        positive(seed, "evaluation seed", integer=True, zero=True)
    require(c["acceptance"] == "strict_improvement" and c["insertion"] == "action_order_existing_first", "unsupported mechanics")
    if c["kind"] == "synthetic":
        require(c["customers"] <= 12 and t["steps"] <= 4 and t["batch_size"] <= 4 and t["rollouts"] <= 4 and e["iterations"] <= 10 and e["seconds"] <= 1 and e["repeats"] <= 4 and len(e["seeds"]) <= 3, "synthetic cap exceeded")
        require(m["width"] <= 64 and m["layers"] <= 2 and t["threads"] <= 2, "synthetic model/thread cap exceeded")


def validate_inputs(c, data, states=None):
    validate_config(c)
    validate_dataset(data)
    ids = {i["id"]: i for i in data}
    families = {}
    for inst in data:
        require(len(inst["demands"]) == c["customers"] + 1 and inst["distance"] == c["distance"], "instance size/distance mismatch")
        require(inst.get("distribution") == c["distribution"], "distribution identity mismatch")
        family = inst.get("family_id", inst["id"])
        require(family not in families or families[family] == inst["split"], "instance family crosses split")
        families[family] = inst["split"]
        if c["kind"] == "synthetic":
            require(inst.get("synthetic") is True, "synthetic flag required on every fixture")
    if c["kind"] == "synthetic":
        require(len(data) <= 12, "synthetic dataset cap exceeded")
    if states is None:
        return
    require(isinstance(states, list) and states, "explicit starting states required")
    if c["kind"] == "synthetic":
        require(len(states) <= 48, "synthetic state cap exceeded")
    seen = set()
    for state in states:
        require(set(state) == {"instance_id", "instance_hash", "routes"}, "state fields must not contain references")
        inst = ids[state["instance_id"]]
        require(instance_hash(inst) == state["instance_hash"], "state instance hash mismatch")
        validate_routes(inst, state["routes"])
        h = digest(state)
        require(h not in seen, "duplicate state")
        seen.add(h)
    require({s["instance_id"] for s in states} == set(ids), "every instance requires a state")


def labels_for(c, data, references):
    training = {i["id"]: i for i in data if i["split"] == "train"}
    require(set(references) == set(training), "only and all training reference IDs required")
    return {key: build_relations(inst, references[key], c["reference"]) for key, inst in training.items()}


def gate(c, synthetic, confirmed):
    if synthetic:
        require(c["kind"] == "synthetic", "synthetic switch cannot run experiment data")
    else:
        require(c["kind"] == "experiment" and confirmed and c.get("budgets_confirmed") is True, "formal execution requires kind=experiment, budgets_confirmed=true and --confirm-real-run")


def identities(c, data, states, references):
    return {"config": digest(c), "data": digest(data), "states": digest(states),
            "references": digest(references) if references is not None else None,
            "code": digest(code_identity())}


def dependency_status():
    return {"torch_available": importlib.util.find_spec("torch") is not None}


def write_json(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Run:
    def __init__(self, path, command, identity, config, parent=None):
        self.path = Path(path)
        require(self.path.parent.is_dir(), "run parent must exist")
        self.path.mkdir(exist_ok=False)
        self.manifest = {"schema": 2, "status": "running", "command": command, "identity": identity,
                         "config": config, "parent": parent, "code_files": code_identity(),
                         "environment": {"python": sys.version, "platform": platform.platform()},
                         "started_unix": time.time(), "attempted_step": 0, "committed_step": 0}
        self.save()

    def save(self):
        temporary = self.path / "manifest.pending.json"
        write_json(temporary, self.manifest, exclusive=False)
        os.replace(temporary, self.path / "manifest.json")

    def metric(self, value):
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def finish(self, status="complete", error=None):
        self.manifest.update(status=status, ended_unix=time.time())
        if error:
            self.manifest["error"] = str(error)
        self.save()
