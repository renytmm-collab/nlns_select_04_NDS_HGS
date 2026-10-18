"""Single-step REINFORCE; every rollout repaired once, no long-horizon returns."""
import math
import os
import random
import time
from pathlib import Path

import torch

from .contracts import require, shuffled_relation
from .neural import create_model, relation_objective
from .runtime import read_json
from .vrp import distances, leave_one_out, ordinary_state, repair, route_cost


def configure_torch(config):
    torch.set_num_threads(config["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    if config["training"]["device"] == "cuda":
        require(torch.cuda.is_available(), "configured CUDA unavailable")
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting CUDA process")
    return config["training"]["device"]


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().square().sum().item()) for g in grads if g is not None))


def batch_loss(model, batch, labels, config, policy_rng):
    policy_losses, auxiliary_losses, rewards_all = [], [], []
    details = []
    for instance, routes in batch:
        h = model.encoder(ordinary_state(instance, routes))
        distance = distances(instance)
        old = route_cost(routes, distance)
        rewards, logs = [], []
        for _ in range(config["training"]["rollouts"]):
            actions, logprob, _ = model.decode(h, config["destroy_size"], policy_rng)
            rebuilt = repair(instance, routes, actions, distance)
            new = route_cost(rebuilt, distance)
            rewards.append((old-new) / max(old, 1e-12))
            logs.append(logprob)
            details.append({"instance_id": instance["id"], "actions": actions, "old_cost": old, "new_cost": new})
        advantages = torch.tensor(leave_one_out(rewards), dtype=h.dtype, device=h.device)
        policy_losses.append(-(advantages.detach() * torch.stack(logs)).mean())
        rewards_all.extend(rewards)
        if config["lambda"] > 0:
            target = labels[instance["id"]]["p"]
            if config["group"] == "shuffled_relation":
                target, _ = shuffled_relation(target, labels[instance["id"]]["instance_hash"], config["shuffle_seed"])
            auxiliary_losses.append(relation_objective(model, h, target))
    policy = torch.stack(policy_losses).mean()
    auxiliary = torch.stack(auxiliary_losses).mean() if auxiliary_losses else policy.new_zeros(())
    total = policy + config["lambda"] * auxiliary if config["lambda"] > 0 else policy
    return total, policy, auxiliary, rewards_all, details


class Trainer:
    def __init__(self, config, data, states, labels, identity):
        self.config, self.labels, self.identity = config, labels, identity
        self.device = configure_torch(config)
        self.model = create_model(config["model"], config["seed"], self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.instances = {i["id"]: i for i in data if i["split"] == "train"}
        require(self.instances, "training split empty")
        require(config["training"]["batch_size"] <= len(self.instances), "batch exceeds unique training instances")
        self.states = {key: [s["routes"] for s in states if s["instance_id"] == key] for key in self.instances}
        self.data_rng = random.Random(config["seed"] + 10001)
        self.policy_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 20001)
        self.order = sorted(self.instances)
        self.data_rng.shuffle(self.order)
        self.cursor = self.epoch = self.step = 0
        self.cumulative_seconds = 0.

    def batch(self):
        if self.cursor == len(self.order):
            self.data_rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        keys = self.order[self.cursor:self.cursor+self.config["training"]["batch_size"]]
        self.cursor += len(keys)
        return [(self.instances[key], self.data_rng.choice(self.states[key])) for key in keys]

    def update(self):
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, policy, auxiliary, rewards, details = batch_loss(self.model, self.batch(), self.labels, self.config, self.policy_rng)
        shared = list(self.model.encoder.parameters())
        pg = grad_norm(policy, shared)
        ag = grad_norm(auxiliary, shared) if self.config["lambda"] > 0 else 0.
        require(bool(torch.isfinite(total).item()), "nonfinite training loss")
        total.backward()
        total_grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.cumulative_seconds += elapsed
        self.step += 1
        return {"step": self.step, "epoch": self.epoch, "policy_loss": float(policy.detach()),
                "relation_loss": float(auxiliary.detach()), "total_loss": float(total.detach()),
                "policy_shared_grad_norm": pg, "relation_shared_grad_norm": ag,
                "weighted_relation_shared_grad_norm": self.config["lambda"] * ag,
                "total_grad_before_clip": float(total_grad), "mean_relative_reward": sum(rewards)/len(rewards),
                "improvement_probability": sum(r > 0 for r in rewards)/len(rewards),
                "rollouts": len(rewards), "update_seconds": elapsed,
                "rollouts_per_second": len(rewards)/max(elapsed, 1e-12), "rollout_details": details}

    def payload(self):
        return {"format": 1, "identity": self.identity, "spec": self.model.spec,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "step": self.step, "order": self.order, "cursor": self.cursor, "epoch": self.epoch,
                "data_rng": self.data_rng.getstate(), "policy_rng": self.policy_rng.get_state(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device == "cuda" else [],
                "cumulative_seconds": self.cumulative_seconds, "torch_version": str(torch.__version__)}

    def restore(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["identity"] == self.identity, "resume identity mismatch (config/data/states/references/code)")
        require(checkpoint["torch_version"] == str(torch.__version__), "resume torch version mismatch")
        parent = read_json(Path(path).parent / "manifest.json")
        require(parent["attempted_step"] == parent["committed_step"] == checkpoint["step"], "resume requires latest committed checkpoint; attempted work cannot be silently refunded")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(self.device)
        self.step, self.cursor, self.epoch = checkpoint["step"], checkpoint["cursor"], checkpoint["epoch"]
        self.order, self.cumulative_seconds = checkpoint["order"], checkpoint["cumulative_seconds"]
        self.data_rng.setstate(checkpoint["data_rng"])
        self.policy_rng.set_state(checkpoint["policy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    def checkpoint(self, directory):
        target = Path(directory) / ("checkpoint-%06d.pt" % self.step)
        require(not target.exists(), "checkpoint already exists")
        pending = target.with_suffix(".pending")
        with pending.open("xb") as stream:
            torch.save(self.payload(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
        return target


def train(config, data, states, labels, identity, run, resume=None, stop_after=None):
    trainer = Trainer(config, data, states, labels, identity)
    if resume:
        trainer.restore(resume)
    require(trainer.step < config["training"]["steps"], "training budget already exhausted")
    run.manifest["environment"].update(torch=str(torch.__version__), device=trainer.device,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if trainer.device == "cuda" else None)
    run.manifest.update(attempted_step=trainer.step, committed_step=trainer.step)
    run.save()
    trainer.checkpoint(run.path)
    started = time.perf_counter()
    limit = min(stop_after, config["training"]["steps"]) if stop_after is not None else config["training"]["steps"]
    require(limit > trainer.step, "stop_after must exceed restored progress")
    for _ in range(trainer.step, limit):
        run.manifest["attempted_step"] = trainer.step + 1
        run.save()
        metrics = trainer.update()
        checkpoint = trainer.checkpoint(run.path)
        run.metric(metrics)
        run.manifest.update(committed_step=trainer.step, latest_checkpoint=checkpoint.name,
                            session_seconds=time.perf_counter()-started)
        run.save()
    run.finish("complete" if trainer.step == config["training"]["steps"] else "interrupted_at_committed_step")
    return {"step": trainer.step, "checkpoint": str(checkpoint), "status": run.manifest["status"]}
