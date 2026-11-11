"""Independent exact coalition best-response stress test, using stdlib only.

Reads exact_diagnostic.json. Does not train a policy or modify the learning experiment.
Main comparison: zero rebate, stationary hidden market, fully informed myopic
grand coalition, exact expected-utility BR; pessimistic seller tie-breaking.
Reports are public, so Bayes adaptation can exploit market information even
though the coalition remains fully coordinated throughout the episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def frac(value):
    return value if isinstance(value, F) else F(str(value))


def encoded(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    if isinstance(value, dict):
        return {str(k): encoded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(v) for v in value]
    return value


@dataclass(frozen=True)
class Market:
    name: str
    demands: tuple[int, ...]
    values: tuple[F, ...]


class ExactMarket:
    def __init__(self, config):
        self.capacity = tuple(config["physics"]["vm_capacities"])
        self.n = config["physics"]["buyers"]
        self.duration = frac(config["physics"]["slot_hours"])
        self.price = frac(config["mechanism"]["price_per_core_hour"])
        self.alpha = frac(config["mechanism"]["alpha"])
        self.cost = frac(config["budget"]["operating_cost_per_core_hour"])
        self.rhos = tuple(map(frac, config["mechanism"]["rho_menu"]))
        self.kappas = tuple(map(frac, config["mechanism"]["kappa_menu"]))
        self.reports = tuple(product(range(config["physics"]["maximum_report"] + 1), repeat=self.n))
        self.markets = tuple(Market(m["id"], tuple(m["m"]), tuple(map(frac, m["v"])))
                             for m in config["hidden_market_catalog"])

    @lru_cache(None)
    def allocations(self, report, rho):
        """Enumerate buyer->VM maps, independently of the prior micro checker."""
        feasible = []
        for assigned in product(range(-1, len(self.capacity)), repeat=self.n):
            leased = [j for j in assigned if j >= 0]
            if len(leased) != len(set(leased)):
                continue
            if any(j >= 0 and not (0 < report[i] <= self.capacity[j]
                                   and report[i] >= rho * self.capacity[j])
                   for i, j in enumerate(assigned)):
                continue
            feasible.append((sum(self.capacity[j] for j in leased), assigned))
        maximum = max(q for q, _ in feasible)
        return tuple(a for q, a in feasible if q == maximum)

    @lru_cache(None)
    def job_value(self, market, leased):
        best = F(0)
        for packing in product((-1,) + leased, repeat=self.n):
            if any(sum(market.demands[i] for i, jj in enumerate(packing) if jj == j)
                   > self.capacity[j] for j in leased):
                continue
            value = sum((market.values[i] for i, j in enumerate(packing) if j >= 0), F(0))
            best = max(best, value)
        return best

    @lru_cache(None)
    def outcome(self, market, rho, kappa, report):
        allocations = self.allocations(report, rho)
        totals = {k: F(0) for k in ("utility", "revenue", "gross_fee", "rebate", "core_hours", "service_value")}
        for assigned in allocations:
            leased = tuple(sorted(j for j in assigned if j >= 0))
            fee = sum((self.price * self.duration *
                       (self.alpha * self.capacity[j] + (1 - self.alpha) * report[i])
                       for i, j in enumerate(assigned) if j >= 0), F(0))
            rebate = sum((kappa * self.price * self.duration * report[i]
                          for i, j in enumerate(assigned) if j >= 0), F(0))
            quantity = self.duration * sum(self.capacity[j] for j in leased)
            value = self.job_value(market, leased)
            totals["utility"] += value - fee + rebate
            totals["revenue"] += fee - rebate - self.cost * quantity
            totals["gross_fee"] += fee
            totals["rebate"] += rebate
            totals["core_hours"] += quantity
            totals["service_value"] += value
        return {k: v / len(allocations) for k, v in totals.items()}

    @lru_cache(None)
    def best_response(self, market, rho, kappa=F(0)):
        outcomes = {b: self.outcome(market, rho, kappa, b) for b in self.reports}
        best_utility = max(o["utility"] for o in outcomes.values())
        best = tuple(b for b in self.reports if outcomes[b]["utility"] == best_utility)
        minimum_revenue = min(outcomes[b]["revenue"] for b in best)
        maximum_revenue = max(outcomes[b]["revenue"] for b in best)
        pessimistic = tuple(b for b in best if outcomes[b]["revenue"] == minimum_revenue)
        mean = {k: sum((outcomes[b][k] for b in pessimistic), F(0)) / len(pessimistic)
                for k in next(iter(outcomes.values()))}
        return {
            "rho": rho, "kappa": kappa,
            "coalition_best_utility": best_utility,
            "utility_maximizing_reports": best,
            "pessimistic_reports": pessimistic,
            "pessimistic_response_distribution": [{"report": b, "probability": F(1, len(pessimistic))}
                                                    for b in pessimistic],
            "pessimistic_mean": mean,
            "best_response_revenue_interval": [minimum_revenue, maximum_revenue],
        }


class ExactBayesPlanner:
    """Finite-horizon planning over public reports; no latent truth is observed.

Primary planner uses kappa=0. Each market draws reports uniformly from its
seller-pessimistic expected-utility maximizers. Conditional on reports and menu,
the random allocation law is identical across markets, so reports suffice for
Bayesian updating here. This tie model is an assumption, not empirical evidence.
"""
    def __init__(self, env):
        self.env = env
        self.prior = tuple(F(1, len(env.markets)) for _ in env.markets)

    @lru_cache(None)
    def branches(self, belief, rho):
        likelihoods = {}
        for idx, market in enumerate(self.env.markets):
            reports = self.env.best_response(market, rho)["pessimistic_reports"]
            for report in reports:
                likelihoods.setdefault(report, [F(0)] * len(belief))[idx] = F(1, len(reports))
        result = []
        for report, likelihood in likelihoods.items():
            probability = sum((b * ell for b, ell in zip(belief, likelihood)), F(0))
            if probability:
                posterior = tuple(b * ell / probability for b, ell in zip(belief, likelihood))
                result.append((report, probability, posterior))
        assert sum((p for _, p, _ in result), F(0)) == 1
        return tuple(result)

    def immediate(self, belief, rho, metric="revenue"):
        return sum((weight * self.env.best_response(m, rho)["pessimistic_mean"][metric]
                    for weight, m in zip(belief, self.env.markets)), F(0))

    @lru_cache(None)
    def solve(self, remaining, belief):
        if remaining == 0:
            return F(0), self.env.rhos[0]
        options = []
        for rho in self.env.rhos:
            value = self.immediate(belief, rho)
            value += sum((prob * self.solve(remaining - 1, posterior)[0]
                          for _, prob, posterior in self.branches(belief, rho)), F(0))
            options.append((value, rho))
        best = max(value for value, _ in options)
        return best, min(rho for value, rho in options if value == best)


def run(config):
    env = ExactMarket(config)
    horizon = config["horizon"]["intervention_rounds"] + config["horizon"]["hold_rounds"]
    planner = ExactBayesPlanner(env)
    stage = {m.name: [env.best_response(m, rho) for rho in env.rhos] for m in env.markets}
    fixed = [{"rho": rho,
              "mean_revenue_per_round": planner.immediate(planner.prior, rho),
              "mean_core_hours_per_round": planner.immediate(planner.prior, rho, "core_hours")}
             for rho in env.rhos]
    fixed_value = max(r["mean_revenue_per_round"] for r in fixed)
    fixed_rho = min(r["rho"] for r in fixed if r["mean_revenue_per_round"] == fixed_value)
    oracle_rows = []
    for market in env.markets:
        values = [(env.best_response(market, rho)["pessimistic_mean"]["revenue"], rho) for rho in env.rhos]
        best = max(value for value, _ in values)
        oracle_rows.append({"market": market.name, "revenue_per_round": best,
                            "optimal_rhos": [rho for value, rho in values if value == best]})
    oracle_mean = sum((row["revenue_per_round"] for row in oracle_rows), F(0)) / len(oracle_rows)
    dynamic_value, probe = planner.solve(horizon, planner.prior)
    # Independently compute a one-probe then fixed exploitation policy.
    by_market = []
    for market in env.markets:
        probe_response = env.best_response(market, probe)
        observations = probe_response["pessimistic_reports"]
        continuation_revenue = F(0)
        continuation_volume = F(0)
        continuation_actions = set()
        for observation in observations:
            posterior = next(post for obs, _, post in planner.branches(planner.prior, probe) if obs == observation)
            best = max(planner.immediate(posterior, rho) for rho in env.rhos)
            rho = min(rho for rho in env.rhos if planner.immediate(posterior, rho) == best)
            continuation_actions.add(rho)
            response = env.best_response(market, rho)["pessimistic_mean"]
            continuation_revenue += response["revenue"] / len(observations)
            continuation_volume += response["core_hours"] / len(observations)
        by_market.append({
            "market": market.name,
            "total_revenue": probe_response["pessimistic_mean"]["revenue"] + (horizon - 1) * continuation_revenue,
            "total_core_hours": probe_response["pessimistic_mean"]["core_hours"] + (horizon - 1) * continuation_volume,
            "post_probe_rhos": sorted(continuation_actions),
        })
    one_probe_value = sum((r["total_revenue"] for r in by_market), F(0)) / len(by_market)
    one_probe_volume = sum((r["total_core_hours"] for r in by_market), F(0)) / len(by_market)
    assert one_probe_value <= dynamic_value <= horizon * oracle_mean
    assert one_probe_value == dynamic_value, "The simple probe policy is not optimal for this config; do not call it optimal."
    assert dynamic_value >= horizon * fixed_value
    # A forward-looking coalition need not reveal its myopic response at the
    # information probe. Restrict to on-support reports, so no unspecified
    # off-model fallback is needed for this counterexample.
    manipulation_rows = []
    probe_branches = planner.branches(planner.prior, probe)
    for market in env.markets:
        candidates = []
        for report, _, posterior in probe_branches:
            post_best = max(planner.immediate(posterior, rho) for rho in env.rhos)
            rho = min(rho for rho in env.rhos if planner.immediate(posterior, rho) == post_best)
            first = env.outcome(market, probe, F(0), report)
            continuation = env.best_response(market, rho)
            later = continuation["pessimistic_mean"]
            future_support = {obs for obs, _, _ in planner.branches(posterior, rho)}
            compatible = all(b in future_support for b in continuation["pessimistic_reports"])
            candidates.append({
                "first_report": report, "induced_posterior": posterior, "induced_fixed_rho": rho,
                "coalition_total_utility": first["utility"] + (horizon - 1) * later["utility"],
                "platform_total_revenue": first["revenue"] + (horizon - 1) * later["revenue"],
                "total_core_hours": first["core_hours"] + (horizon - 1) * later["core_hours"],
                "subsequent_reports_remain_on_model_support": compatible,
            })
        best_u = max(c["coalition_total_utility"] for c in candidates)
        optimal = [c for c in candidates if c["coalition_total_utility"] == best_u]
        worst = min(optimal, key=lambda c: (c["platform_total_revenue"], c["first_report"]))
        manipulation_rows.append({"market": market.name, **worst})
    manipulation_revenue = sum((r["platform_total_revenue"] for r in manipulation_rows), F(0)) / len(env.markets)
    manipulation_volume = sum((r["total_core_hours"] for r in manipulation_rows), F(0)) / len(env.markets)
    # Strong-seller tie intervals reveal fragility to indifference and rebates.
    rebate_diagnostic = {m.name: [env.best_response(m, rho, kappa)
                                for rho in env.rhos for kappa in env.kappas
                                if kappa <= 1 - env.alpha]
                         for m in env.markets}
    return {
        "status": "completed_exact_memoryless_grand_coalition_stress_test",
        "assumptions": {
            "price": env.price, "alpha": env.alpha, "duration": env.duration,
            "capacity": env.capacity, "zero_rebates_in_primary_comparison": True,
            "stationary_market_per_episode": True,
            "types_known_to_coalition_not_platform": True,
            "allocation_ties": "uniform over maximum rented capacity matchings",
            "coalition_ties": "minimum expected platform net revenue, then uniform among tied reports",
            "coalition_is_myopic_exact_best_response": True,
            "coalition_remains_intact_every_round": True,
            "platform_observation_for_bayes": "public joint report only",
            "primary_market_prior": planner.prior,
            "random_allocation_adds_no_type_information_conditional_on_report": True,
            "operating_cost_per_core_hour": env.cost,
        },
        "stage_best_responses": stage,
        "fixed_policy_uniform_market_prior": fixed,
        "continuous_threshold_regimes_for_capacity_3_2": {
            "scope": "For capacities (3,2) and positive reports 1..3, the menu contains a representative of every possible eligibility regime.",
            "regimes": [
                {"interval": "[0,1/3]", "representative": F(0)},
                {"interval": "(1/3,1/2]", "representative": F(1, 2)},
                {"interval": "(1/2,2/3]", "representative": F(2, 3)},
                {"interval": "(2/3,1]", "representative": F(4, 5)},
            ],
            "rho_4_5_and_1_are_economically_identical": True,
        },
        "per_market_oracle_fixed": oracle_rows,
        "horizon_comparison": {
            "rounds": horizon,
            "best_common_fixed_rho": fixed_rho,
            "best_common_fixed_total_revenue": horizon * fixed_value,
            "per_market_oracle_total_revenue": horizon * oracle_mean,
            "bayes_optimal_total_revenue": dynamic_value,
            "bayes_optimal_first_rho": probe,
            "one_probe_then_fixed_total_revenue": one_probe_value,
            "one_probe_then_fixed_total_core_hours": one_probe_volume,
            "one_probe_then_fixed_by_market": by_market,
            "improvement_over_common_fixed": dynamic_value - horizon * fixed_value,
            "relative_improvement_over_common_fixed": dynamic_value / (horizon * fixed_value) - 1,
            "regret_to_per_market_oracle": horizon * oracle_mean - dynamic_value,
            "fits_intervention_then_fixed_hold_protocol": True,
            "rebate_spend": F(0),
        },
        "on_support_forward_looking_probe_manipulation": {
            "attack_scope": "exact optimal first-round report among prior-predictive support, followed by exact stage BR to the induced fixed threshold",
            "first_round_report_is_allowed_to_sacrifice_current_utility": True,
            "off_model_reports_or_fallback_not_needed": True,
            "not_claimed_to_exhaust_all_dynamic_attacks": True,
            "by_market": manipulation_rows,
            "mean_total_platform_revenue": manipulation_revenue,
            "mean_total_core_hours": manipulation_volume,
            "difference_from_best_common_fixed": manipulation_revenue - horizon * fixed_value,
            "loss_from_myopic_model_prediction": dynamic_value - manipulation_revenue,
        },
        "rebate_stage_diagnostics_only": rebate_diagnostic,
        "proof_scope": [
            "For a fixed type and stationary myopic exact BR, zero-rebate history-dependent schedules have conditional expected stage revenue <= the maximum stage revenue across rho; summing gives H times that fixed-type oracle ceiling.",
            "Unknown-market observation-dependent adaptation can beat a single common fixed rho by learning which market-specific rho to choose; this does not require breaking a coalition or shaping a learner.",
            "The ceiling does not cover changed action menus, rebates, intertemporal coalition strategies, changing supply/types, or fixed-service constraints absent from the comparison.",
            "The rebate table is a one-stage sensitivity analysis only: it does not apply the episode cash budget or the mandatory no-rebate hold phase and is not a feasible 400-round policy comparison.",
            "Bayesian optimality assumes the declared exact response and tie model; it is not robustness to strategic manipulation of platform beliefs.",
        ],
        "checks": {"independent_enumeration": True, "exact_rational_arithmetic": True,
                   "one_probe_equals_finite_horizon_bayes_DP": one_probe_value == dynamic_value,
                   "oracle_ceiling_holds": dynamic_value <= horizon * oracle_mean},
    }


def write_report(data, output):
    def number(x):
        return f"{float(x):.4f}".rstrip("0").rstrip(".")
    lines = ["# Exact coalition best-response stress test", "",
             "This is a completed exact-model comparison, not a learned-policy experiment. The coalition remains fully coordinated in every round.", "",
             "| Market | rho=0 | rho=1/2 | rho=2/3 | rho=4/5 | rho=1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for market, rows in data["stage_best_responses"].items():
        lines.append("| " + market + " | " + " | ".join(number(r["pessimistic_mean"]["revenue"]) for r in rows) + " |")
    h = data["horizon_comparison"]
    lines.extend(["", f"At horizon {h['rounds']}, the best common fixed threshold is {h['best_common_fixed_rho']}; expected revenue is {number(h['best_common_fixed_total_revenue'])}.",
                  f"One public-response probe at rho={h['bayes_optimal_first_rho']}, followed by a fixed posterior-optimal threshold, obtains {number(h['bayes_optimal_total_revenue'])}. This equals exact finite-horizon Bayesian planning under the declared response model.",
                  f"The fixed-type oracle is {number(h['per_market_oracle_total_revenue'])}; information-acquisition regret is {number(h['regret_to_per_market_oracle'])}. Mean rented core-hours are {number(h['one_probe_then_fixed_total_core_hours'])}, and rebates are zero.", "",
                  "The gain over a common fixed rule is market identification and personalized choice across episodes. It is not evidence of coalition breakup. It does not require a slowly adapting learner, but it does assume myopic exact responses rather than full-horizon strategic reporting.", "",
                  "With a fixed known market and memoryless exact coalition BR, expected revenue in each round is at most that market's best-stage threshold revenue. Randomizing or scheduling thresholds cannot exceed this ceiling under the same zero-rebate menu.", "",
                  "For capacities (3,2) and reports 1..3, the four eligibility regimes are [0,1/3], (1/3,1/2], (1/2,2/3], and (2/3,1]. The tested menu represents all four, so a finer continuous-threshold search cannot improve this exact stationary comparison. rho=4/5 and rho=1 are economically identical here.", "",
                  "Market D is sensitive to indifferent coalition responses. At strict thresholds and zero rebates, the coalition can obtain the same utility by renting only the 3-core VM (revenue 3) or both VMs (revenue 5); the primary result uses the seller-pessimistic choice. A small rebate can remove that indifference, but the rebate-stage table does not incorporate the full-episode budget or hold phase.", "",
                  "The exact-response probe assumes myopic best responses. Exact one-step belief manipulation is also evaluated: the coalition may sacrifice utility in the probe round and maximize its full-horizon surplus using an on-model-support report."])
    attack = data["on_support_forward_looking_probe_manipulation"]
    lines.extend(["", f"Under this forward-looking probe attack, mean platform revenue is {number(attack['mean_total_platform_revenue'])}, versus {number(h['best_common_fixed_total_revenue'])} for the common fixed threshold. High-value A/C can report (0,3), mimic market D, and induce rho=1/2 for the remaining rounds. Their subsequent optimal reports are observationally compatible with D, so this attack does not need an off-model observation or an arbitrary fallback rule.",
                  "This attack is exact within its specified first-round-deviation class and is a lower bound on overall dynamic exploitability. It shows why gains against myopic or bounded-response buyers do not by themselves establish strategic anti-collusion robustness."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.setrecursionlimit(10000)
    config = json.loads((ROOT / "config" / "exact_diagnostic.json").read_text(encoding="utf-8"), parse_float=Decimal)
    data = run(config)
    destination = ROOT / "results"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "exact_threshold_stress.json").write_text(json.dumps(encoded(data), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(data, destination / "exact_threshold_stress.md")
    h = data["horizon_comparison"]
    print("Exact coalition stress test complete.")
    print("Fixed:", h["best_common_fixed_total_revenue"], "Bayes:", h["bayes_optimal_total_revenue"],
          "Oracle:", h["per_market_oracle_total_revenue"], "Core-hours:", h["one_probe_then_fixed_total_core_hours"])
    print("Checks:", data["checks"])


if __name__ == "__main__":
    main()
