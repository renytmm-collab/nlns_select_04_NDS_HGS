import argparse
import json
import platform
from pathlib import Path
import sys

from .contracts import build_relations, digest, require, validate_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(config_path, data_path, reference_path):
    config, instances = read(config_path), read(data_path)
    require(config["schema"] == 1, "unsupported schema")
    validate_dataset(instances)
    references = read(reference_path)
    training = {x["id"]: x for x in instances if x["split"] == "train"}
    require(set(references) == set(training), "reference IDs must match training instances exactly")
    labels = {key: build_relations(inst, references[key], config["reference"]) for key, inst in training.items()}
    return config, instances, references, labels


def main():
    parser = argparse.ArgumentParser(description="Reference contracts; use python -m relation_lab.experiment for experiments")
    parser.add_argument("command", choices=["preflight", "import-references", "generate-references", "train", "evaluate"])
    parser.add_argument("--config")
    parser.add_argument("--data")
    parser.add_argument("--references")
    parser.add_argument("--run-dir")
    parser.add_argument("--confirm-real-run", action="store_true")
    args = parser.parse_args()
    if args.command in ("generate-references", "train", "evaluate"):
        if not args.config or not args.data:
            parser.error("--config and --data required; see python -m relation_lab.experiment --help")
        from .experiment import main as experiment_main
        raise SystemExit(experiment_main(sys.argv[1:]))
    for key in ("config", "data", "references"):
        if not getattr(args, key):
            parser.error("--" + key + " required")
    config, instances, references, labels = preflight(args.config, args.data, args.references)
    if args.command == "preflight":
        print(json.dumps({"contract_valid": True, "training_ready": False, "instances": len(instances), "training_labels": len(labels)}))
        return
    require(args.run_dir is not None, "--run-dir required")
    output = Path(args.run_dir)
    require(output.parent.is_dir(), "run parent must already exist")
    output.mkdir(exist_ok=False)
    # A partial run remains visible on failure; no retry, overwrite or cleanup.
    manifest = {"status": "started", "kind": "reference_import_only", "config": config, "config_hash": digest(config), "data_hash": digest(instances), "reference_hash": digest(references), "python": sys.version, "platform": platform.platform(), "code_hash": digest({p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "run_dir": str(output)}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print("FAILED: " + str(exc), file=sys.stderr)
        sys.exit(1)
