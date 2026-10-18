"""Requires an existing torch installation; never installs or compiles anything."""
import copy
import importlib.util
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(AVAILABLE, "PyTorch absent; neural validation must run on server")
class Neural(unittest.TestCase):
    def setUp(self):
        import torch
        from relation_lab.fixtures import smoke_bundle
        from relation_lab.neural import create_model
        self.torch = torch
        torch.set_num_threads(1)
        self.c, self.data, self.states, self.refs = smoke_bundle()
        self.model = create_model(self.c["model"], self.c["seed"])

    def test_mask_conditional_dependence_and_export(self):
        from relation_lab.neural import export_policy
        from relation_lab.vrp import ordinary_state
        torch = self.torch
        state = ordinary_state(self.data[0], self.states[0]["routes"])
        h = self.model.encoder(state)
        actions, _, distributions = self.model.decode(h, 8, torch.Generator().manual_seed(5))
        self.assertEqual(sorted(actions), list(range(1, 9)))
        for t, logp in enumerate(distributions):
            self.assertTrue(torch.isneginf(logp[0]))
            for node in actions[:t]:
                self.assertTrue(torch.isneginf(logp[node]))
        _, _, a = self.model.decode(h, 3, forced=[1, 2, 3])
        _, _, b = self.model.decode(h, 3, forced=[2, 1, 3])
        self.assertFalse(torch.allclose(a[2][3:], b[2][3:]))
        policy = export_policy(self.model)
        self.assertFalse(any(k.startswith("relation.") for k in policy.state_dict()))
        exported_h = policy.encoder(state)
        self.assertTrue(torch.equal(h, exported_h))
        aa = self.model.decode(h, 3, greedy=True)
        bb = policy.decode(exported_h, 3, greedy=True)
        self.assertEqual(aa[0], bb[0])
        self.assertTrue(torch.equal(aa[1], bb[1]))

    def test_symmetry_shared_gradient_and_input_isolation(self):
        from relation_lab.neural import relation_objective
        from relation_lab.runtime import labels_for
        from relation_lab.vrp import ordinary_state
        torch = self.torch
        state = ordinary_state(self.data[0], self.states[0]["routes"])
        h = self.model.encoder(state)
        logits = self.model.pair_logits(h)
        self.assertTrue(torch.equal(logits, logits.T))
        labels = labels_for(self.c, self.data, self.refs)
        relation_objective(self.model, h, labels[self.data[0]["id"]]["p"]).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoder.parameters() if p.grad is not None), 0)
        with self.assertRaises(ValueError):
            self.model.encoder(dict(state, reference_path="forbidden"))
        reverse = dict(state, routes=[list(reversed(r)) for r in reversed(state["routes"])])
        self.assertTrue(torch.allclose(h, self.model.encoder(reverse), atol=1e-6))

    def test_lambda_zero_and_rng(self):
        from relation_lab.runtime import identities, labels_for
        from relation_lab.training import Trainer, batch_loss
        a = Trainer(self.c, self.data, self.states, {}, identities(self.c, self.data, self.states, None))
        c = copy.deepcopy(self.c)
        c["group"] = "relation"
        b = Trainer(c, self.data, self.states, {}, identities(c, self.data, self.states, None))
        am, bm = a.update(), b.update()
        self.assertEqual(am["rollout_details"], bm["rollout_details"])
        self.assertEqual(am["total_loss"], bm["total_loss"])
        for key, value in a.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, b.model.state_dict()[key]))
        # lambda>0 changes gradients but not the exogenous policy samples before updates.
        c["lambda"] = .1
        labels = labels_for(c, self.data, self.refs)
        d = Trainer(c, self.data, self.states, labels, identities(c, self.data, self.states, self.refs))
        dm = d.update()
        self.assertEqual(am["rollout_details"], dm["rollout_details"])
        self.assertGreater(dm["relation_shared_grad_norm"], 0)

    def test_full_synthetic_lifecycle_and_resume(self):
        from relation_lab.experiment import main
        from relation_lab.runtime import read_json
        torch = self.torch
        root = Path("test_artifacts") / ("neural-"+uuid.uuid4().hex)
        root.mkdir(parents=True)
        fixture = root/"fixture"
        def run(args):
            with patch("sys.stdout"):
                self.assertEqual(main(args), 0, str(args))
        run(["make-smoke", "--run-dir", str(fixture)])
        base = ["--config", str(fixture/"relation.json"), "--data", str(fixture/"data.json"), "--states", str(fixture/"states.json")]
        train_args = base+["--references", str(fixture/"references.json"), "--synthetic"]
        run(["train", *train_args, "--run-dir", str(root/"full")])
        run(["train", *train_args, "--stop-after", "1", "--run-dir", str(root/"partial")])
        run(["resume", *train_args, "--checkpoint", str(root/"partial/checkpoint-000001.pt"), "--run-dir", str(root/"resumed")])
        a = torch.load(root/"full/checkpoint-000002.pt", weights_only=True)
        b = torch.load(root/"resumed/checkpoint-000002.pt", weights_only=True)
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        self.assertTrue(torch.equal(a["policy_rng"], b["policy_rng"]))
        self.assertEqual(a["data_rng"], b["data_rng"])
        self.assertEqual(a["cursor"], b["cursor"])
        self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
        for parameter, saved in a["optimizer"]["state"].items():
            for key, value in saved.items():
                self.assertTrue(torch.equal(value, b["optimizer"]["state"][parameter][key]))
        run(["export", "--checkpoint", str(root/"full/checkpoint-000002.pt"), "--run-dir", str(root/"export")])
        policy = root/"export/policy.pt"
        # Reference source is deliberately removed from the fixture name space.
        (fixture/"references.json").rename(fixture/"references.offline.json")
        run(["deploy", *base, "--policy", str(policy), "--synthetic", "--run-dir", str(root/"deploy")])
        run(["evaluate", *base, "--policy", str(policy), "--synthetic", "--split", "validation", "--run-dir", str(root/"validation")])
        run(["freeze", *base, "--policy", str(policy), "--validation-results", str(root/"validation/results.json"), "--run-dir", str(root/"freeze")])
        run(["evaluate", *base, "--policy", str(policy), "--synthetic", "--split", "test", "--freeze", str(root/"freeze/freeze.json"), "--run-dir", str(root/"test")])
        results = read_json(root/"test/results.json")
        self.assertEqual(set(results["summary"]), {"single", "search_iterations", "search_time"})

    def test_auxiliary_normalization_and_shuffle_stream(self):
        from relation_lab.runtime import labels_for
        from relation_lab.training import batch_loss
        c = copy.deepcopy(self.c)
        c.update(group="relation")
        c["lambda"] = .1
        labels = labels_for(c, self.data, self.refs)
        batch = [(self.data[0], self.states[0]["routes"])]
        a = batch_loss(self.model, batch, labels, c, self.torch.Generator().manual_seed(101))
        c["training"]["rollouts"] = 4
        b = batch_loss(self.model, batch, labels, c, self.torch.Generator().manual_seed(101))
        self.assertTrue(self.torch.equal(a[2], b[2]))
        c["group"] = "shuffled_relation"
        d = batch_loss(self.model, batch, labels, c, self.torch.Generator().manual_seed(101))
        self.assertEqual(b[4], d[4])
        self.assertTrue(self.torch.isfinite(d[2]))


if __name__ == "__main__":
    unittest.main()
