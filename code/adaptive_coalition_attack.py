"""Restricted forward-looking coalition attacks on frozen learned platforms.

Predeclared platform seeds: 0,4,9. Attack selection uses 400-round coalition
utility on separate validation trajectories, never platform revenue. Final
testing uses fresh matching/report random streams. No platform file is edited.
The search is finite and gives a lower bound on dynamic exploitability; absence
of a successful template is not a robustness certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from steering_env import BatchMarket, JOINT_REPORTS, TABLES, RHO_VALUES
from belief_baselines import BeliefTracker
from run_threshold_experiment import (
    Policy, public_features, run_episodes, PRIMARY, BLOCK, CONTROL,
    HORIZON, MACROS, CONTROL_MACROS,
)

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "experiments" / "dynamic_threshold"
OUT = ROOT / "experiments" / "adaptive_coalition_attack"
PLATFORM_SEEDS = (0, 4, 9)
VALIDATION_WORLDS = 8
TEST_WORLDS = 64
VALIDATION_SEED = 271828
TEST_SEED = 314159
METRICS = ("platform_revenue", "coalition_utility", "core_hours", "completed_value", "completed_jobs")


@dataclass(frozen=True)
class Template:
    name: str
    family: str
    prefix: int = 0
    report_index: int = -1
    fake_market: int = -1
    fake_model: int = -1


def search_space():
    result = [Template("exact_myopic_BR", "exact")]
    for length in (10, 40, 100, 200):
        for report in range(16):
            b = JOINT_REPORTS[report]
            result.append(Template(f"constant_{int(b[0])}{int(b[1])}_first_{length}",
                                   "constant_prefix", length, report_index=report))
    for length in (40, 200):
        for market in range(4):
            result.append(Template(f"exact_market_{'ABCD'[market]}_first_{length}",
                                   "foreign_exact_prefix", length, fake_market=market))
    for model in range(16):
        result.append(Template(f"emulate_learning_model_{model}_all_400", "emulate_model", HORIZON,
                               fake_model=model))
    assert len(result) == 89
    return tuple(result)


def exact_report_table():
    """Only the coalition's own value/type enters its instantaneous oracle."""
    table = np.zeros((4, len(RHO_VALUES)), dtype=np.int64)
    for market in range(4):
        for action in range(len(RHO_VALUES)):
            utility = np.zeros(16)
            revenue = np.zeros(16)
            for joint in range(16):
                n = TABLES["counts"][action, joint]
                utility[joint] = TABLES["utility"][market, 1, action, joint, :n].sum(axis=-1).mean()
                revenue[joint] = TABLES["revenue"][action, joint, :n].mean()
            utility_best = utility >= utility.max() - 1e-10
            worst_revenue = revenue[utility_best].min()
            eligible = utility_best & (revenue <= worst_revenue + 1e-10)
            table[market, action] = np.flatnonzero(eligible)[0]
    return table


EXACT_REPORT = exact_report_table()


def simulate(policy, market, templates, worlds, stream_seed):
    """All templates share random streams by world, not by chosen reports.

    The attacker observes the announced rule and time. An emulated learner is
    attacker-owned hypothetical state; it is never the platform's belief. The
    platform receives the usual public features and updates its own posterior
    on the attacked trajectory. All actual accounts belong to one coalition.
    """
    count = len(templates)
    batch = count * worlds
    template_rows = np.repeat(np.arange(count), worlds)
    true_ids = np.full(batch, 4 * market + 2, dtype=np.int64)
    # Actual learner weights are unused by the attack, but the environment's
    # physical/accounting path is preserved by forced reports.
    env = BatchMarket(true_ids, learner_mode="canonical")
    tracker = BeliefTracker(batch, learner_mode="canonical")
    fake_rows = np.array([row for row in range(batch)
                          if templates[template_rows[row]].family == "emulate_model"], dtype=int)
    fake_env = None
    if len(fake_rows):
        fake_ids = np.array([templates[template_rows[row]].fake_model for row in fake_rows])
        fake_env = BatchMarket(fake_ids, learner_mode="canonical")
    rng = np.random.default_rng(stream_seed)
    totals = np.zeros((batch, len(METRICS)), dtype=np.float64)
    actions = np.zeros((MACROS, batch), dtype=np.int8)
    report_counts = np.zeros((batch, 16), dtype=np.int64)
    block_metrics = np.zeros((MACROS, batch, 3), dtype=np.float32)
    hidden = None
    last_action = np.zeros(batch, dtype=int)
    obs_np = public_features(last_action, [], 0, tracker.posterior, "belief")
    families = np.array([templates[j].family for j in template_rows])
    prefix = np.array([templates[j].prefix for j in template_rows])
    report_indices = np.array([templates[j].report_index for j in template_rows])
    foreign_market = np.array([templates[j].fake_market for j in template_rows])
    policy.eval()
    for macro in range(MACROS):
        if macro < CONTROL_MACROS:
            with torch.no_grad():
                logits, _, hidden = policy(torch.from_numpy(obs_np)[None], hidden)
            local_action = logits[0].argmax(-1).numpy()
            last_action = local_action.copy()
        else:
            local_action = last_action.copy()
        env_action = PRIMARY[local_action]
        actions[macro] = local_action
        observed = []
        for offset in range(BLOCK):
            t = macro * BLOCK + offset
            u = np.tile(rng.random((worlds, 3)), (count, 1))
            joint = EXACT_REPORT[market, env_action].copy()
            constant = (families == "constant_prefix") & (t < prefix)
            joint[constant] = report_indices[constant]
            foreign = (families == "foreign_exact_prefix") & (t < prefix)
            joint[foreign] = EXACT_REPORT[foreign_market[foreign], env_action[foreign]]
            if fake_env is not None:
                fake_public, _ = fake_env.step(env_action[fake_rows], u[fake_rows])
                joint[fake_rows] = 4 * fake_public["reports"][:, 0] + fake_public["reports"][:, 1]
            outcome = np.floor(u[:, 2] * TABLES["counts"][env_action, joint]).astype(int)
            phi, qi, qg = env._predictions(env_action)
            public, metric = env._realize_and_update(env_action, joint, outcome, phi, qi, qg)
            # No decisions use the posterior after the threshold is frozen.
            # Omitting that unused computation preserves every action/cash flow;
            # buyer/attacker state and physical accounting still update fully.
            if macro < CONTROL_MACROS:
                tracker.observe(env_action, public["reports"], public["owners"])
            observed.append(public)
            increment = np.column_stack((public["revenue"], metric["total_user_utility"],
                                         public["volume"], metric["completed_value"], metric["completed_jobs"]))
            totals += increment
            block_metrics[macro] += increment[:, :3]
            report_counts[np.arange(batch), joint] += 1
        if macro + 1 < CONTROL_MACROS:
            obs_np = public_features(local_action, observed, macro + 1, tracker.posterior, "belief")
    return {
        "metrics": totals.reshape(count, worlds, len(METRICS)),
        "actions": actions.reshape(MACROS, count, worlds),
        "report_counts": report_counts.reshape(count, worlds, 16),
        "block_metrics": block_metrics.reshape(MACROS, count, worlds, 3),
        "posterior_at_end_of_control": tracker.posterior.reshape(count, worlds, 16),
    }


def choose_by_utility(metrics):
    """Selection has no reference to the platform-revenue column."""
    return int(np.argmax(metrics[:, :, METRICS.index("coalition_utility")].mean(axis=1)))


def paired_summary(selected, reference, seed):
    delta = selected - reference
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist()}


def protocol():
    return {
        "status": "predeclared_restricted_attack_search",
        "platform_seeds": list(PLATFORM_SEEDS),
        "platform_mode": "belief", "frozen_network_weights": True,
        "platform_state_updates": "GRU, candidate posterior and inferred learner states update normally; 200-round hold freezes only the last threshold",
        "horizon": HORIZON, "control_rounds": CONTROL, "block_rounds": BLOCK,
        "markets": ["A", "B", "C", "D"], "grand_coalition": True,
        "rebates": 0, "registered_accounts": 2,
        "selection_objective": "mean full-400-round coalition total utility; deterministic lowest template index breaks ties",
        "selection_uses_platform_revenue": False,
        "validation_worlds_per_market_template": VALIDATION_WORLDS,
        "test_worlds_per_market_selected_attack": TEST_WORLDS,
        "validation_stream_base": VALIDATION_SEED, "test_stream_base": TEST_SEED,
        "stream_formula": "base + 10000*platform_seed + 101*market_index",
        "paired_randomness": "same exogenous stream per world across attack templates; validation and test are disjoint",
        "attacker_observations": "own coalition types, current public threshold, time, and attacker-simulated fake learner state",
        "attacker_does_not_observe": "platform private posterior/GRU hidden state or actual privileged metrics at action time",
        "attacker_privileged_offline_access": "white-box frozen policy and simulator for validation search",
        "scope": "finite 89-template search, lower bound on exploitability; not an equilibrium computation or robustness certificate",
        "templates": [asdict(t) for t in search_space()],
    }


def prepare_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    proposed = protocol()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != proposed:
            raise RuntimeError("Existing attack protocol differs; do not silently change a predeclared search.")
    else:
        path.write_text(json.dumps(proposed, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_seed(seed):
    if seed not in PLATFORM_SEEDS:
        raise ValueError("Platform seed was not predeclared")
    protocol_hash = prepare_protocol()
    checkpoint_path = PLATFORM / "checkpoints" / f"belief_{seed}.pt"
    final_log = PLATFORM / "logs" / f"belief_{seed}.json"
    freeze_path = PLATFORM / "pretest_freeze.json"
    if not checkpoint_path.exists() or not final_log.exists() or not freeze_path.exists():
        raise FileNotFoundError(f"Final checkpoint and completion log not ready for seed {seed}")
    training_log = json.loads(final_log.read_text(encoding="utf-8"))
    if training_log[-1]["iteration"] < 39:
        raise RuntimeError("Training log has not completed the declared 39 iterations")
    payload = checkpoint_path.read_bytes()
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    # New manifests use POSIX separators. Accept historical Windows manifests
    # when a saved experiment is inspected on Linux or macOS.
    hashes = {key.replace("\\", "/"): value
              for key, value in freeze_manifest["hashes"].items()}
    expected_hash = hashes[checkpoint_path.relative_to(ROOT).as_posix()]
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise RuntimeError("Checkpoint does not match the frozen pretest manifest")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=False)
    policy = Policy()
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    templates = search_space()
    start = time.perf_counter()
    validation = []
    selections = []
    for market in range(4):
        result = simulate(policy, market, templates, VALIDATION_WORLDS,
                          VALIDATION_SEED + 10000 * seed + 101 * market)
        choice = choose_by_utility(result["metrics"])
        validation.append(result)
        selections.append({"market": "ABCD"[market], "template_index": choice,
                           "template": asdict(templates[choice]),
                           "validation_mean_coalition_utility": float(result["metrics"][choice, :, 1].mean()),
                           "baseline_validation_mean_coalition_utility": float(result["metrics"][0, :, 1].mean())})
        print(json.dumps({"attack_validation_done": seed, "market": "ABCD"[market],
                          "selected": templates[choice].name, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    # Persist and freeze the utility-only choice before opening any test stream.
    selection = {"platform_seed": seed, "checkpoint_sha256": checkpoint_hash,
                 "protocol_sha256": protocol_hash, "selections": selections}
    (OUT / f"selection_{seed}.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"validation_{seed}.npz", metric_names=np.array(METRICS),
                        **{f"market_{i}_{key}": value for i, result in enumerate(validation) for key, value in result.items()})
    test_results = []
    rows = []
    for market, selection_row in enumerate(selections):
        selected = templates[selection_row["template_index"]]
        result = simulate(policy, market, (selected, templates[0]), TEST_WORLDS,
                          TEST_SEED + 10000 * seed + 101 * market)
        attack, reference = result["metrics"]
        row = {"market": "ABCD"[market], "selected_template": asdict(selected),
               "attack_test_means": dict(zip(METRICS, attack.mean(axis=0).tolist())),
               "exact_BR_test_means": dict(zip(METRICS, reference.mean(axis=0).tolist())),
               "coalition_utility_gain": paired_summary(attack[:, 1], reference[:, 1], 9000 + seed * 100 + market),
               "platform_revenue_change": paired_summary(attack[:, 0], reference[:, 0], 9500 + seed * 100 + market),
               "core_hours_change": paired_summary(attack[:, 2], reference[:, 2], 9800 + seed * 100 + market)}
        rows.append(row)
        test_results.append(result)
        print(json.dumps({"attack_test_done": seed, **row}), flush=True)
    np.savez_compressed(OUT / f"test_{seed}.npz", metric_names=np.array(METRICS),
                        **{f"market_{i}_{key}": value for i, result in enumerate(test_results) for key, value in result.items()})
    summary = {
        "status": "completed_validation_selected_restricted_attack",
        "platform_seed": seed, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selected_iteration": checkpoint.get("selected_iteration"),
        "protocol_sha256": protocol_hash, "markets": rows,
        "pretest_freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "equivalent_hold_optimization": "After round 200, skip unused posterior/GRU computation; threshold is frozen and all buyer/attacker/physical updates continue. Saved posterior is labeled end-of-control, not end-of-episode.",
        "uniform_market_mean_attack_revenue": float(np.mean([r["attack_test_means"]["platform_revenue"] for r in rows])),
        "uniform_market_mean_reference_revenue": float(np.mean([r["exact_BR_test_means"]["platform_revenue"] for r in rows])),
        "uniform_market_mean_utility_gain": float(np.mean([r["coalition_utility_gain"]["mean"] for r in rows])),
        "seconds": time.perf_counter() - start,
        "limitations": [
            "Only the 89 predeclared templates were searched; unsuccessful search is not a robustness guarantee.",
            "Selection used eight validation trajectories per type and may overfit; all reported effects use independent 64-world tests.",
            "Three platform seeds were preselected; these are not all ten training seeds.",
            "Report emulation assumes the coalition knows its own types and can coordinate randomized reports; it does not observe the platform's private belief.",
            "Inference intervals resample matching/report worlds within a fixed trained policy and type, not the full training-seed distribution.",
        ],
    }
    (OUT / f"summary_{seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def self_test():
    torch.manual_seed(1729)
    policy = Policy().eval()
    baseline = search_space()[0]
    result = simulate(policy, 2, (baseline,), 3, 7717)
    reference = run_episodes("belief", np.full(3, 10), 7717, policy=policy, response="exact_grand")
    for column, key in ((0, "revenue"), (1, "total_user_utility"), (2, "volume"),
                        (3, "completed_value"), (4, "completed_jobs")):
        np.testing.assert_allclose(result["metrics"][0, :, column], reference["totals"][key], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result["actions"][:, 0], reference["actions"])
    fake = Template("selftest_emulation", "emulate_model", HORIZON, fake_model=7)
    emulated = simulate(policy, 0, (fake,), 3, 12317)
    reference_fake = run_episodes("belief", np.full(3, 7), 12317, policy=policy, response="learning")
    np.testing.assert_allclose(emulated["metrics"][0, :, 0], reference_fake["totals"]["revenue"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(emulated["actions"][:, 0], reference_fake["actions"])
    # Changing the platform-revenue column cannot change an attack selection.
    artificial = np.zeros((3, 5, len(METRICS)))
    artificial[1, :, 1] = 4
    assert choose_by_utility(artificial) == 1
    artificial[:, :, 0] = np.array([999999, -999999, 1000000])[:, None]
    assert choose_by_utility(artificial) == 1
    assert len(search_space()) == 89
    print("PASS: forced exact response equals root evaluator; fake-model emulation public path replay; utility-only attack selection; fixed 89-template search.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, choices=PLATFORM_SEEDS)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--deadline-minutes", type=float, default=180)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.prepare or args.wait or args.seed is not None:
        print("Protocol SHA256:", prepare_protocol(), flush=True)
    seeds = (args.seed,) if args.seed is not None else PLATFORM_SEEDS
    if args.wait:
        pending = [s for s in seeds if not (OUT / f"summary_{s}.json").exists()]
        deadline = time.monotonic() + args.deadline_minutes * 60
        while pending:
            ready = [s for s in pending if (PLATFORM / "logs" / f"belief_{s}.json").exists()
                     and (PLATFORM / "checkpoints" / f"belief_{s}.pt").exists()
                     and (PLATFORM / "pretest_freeze.json").exists()]
            for seed in ready:
                # Fresh imports after the freeze avoid retaining a module that
                # was still being edited when this waiting process started.
                completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                            "--seed", str(seed)],
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if completed.returncode:
                    raise RuntimeError(f"Attack evaluation failed for seed {seed}")
                pending.remove(seed)
            if pending:
                if time.monotonic() >= deadline:
                    print(json.dumps({"pending_platform_seeds": pending, "status": "wait_deadline"}), flush=True)
                    return
                time.sleep(10)
    elif args.seed is not None:
        execute_seed(args.seed)


if __name__ == "__main__":
    main()
