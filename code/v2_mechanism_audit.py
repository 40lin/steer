"""Independent exact audit of reported-demand and delivered-capacity billing.

All payoffs and matching probabilities use fractions. This module does not
modify v1 tables, train a controller, or look at learned-controller outcomes.
The capacity-billing conclusions are conditional on the four declared markets,
free coalition sharing, fixed registered identities and a committed unit price.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'v2_audit'
CAPACITY = (3, 2)
MARKETS = (
    ('A', (3, 2), (F(9), F(5))),
    ('B', (3, 2), (F('2.4'), F('1.6'))),
    ('C', (2, 2), (F(5), F(5))),
    ('D', (1, 3), (F(2), F(8))),
)
RHOS = (F(0), F(1, 2), F(2, 3), F(1))
PRICE_GRID = tuple(map(F, ('.65', '.68', '.70', '.72', '.75', '.78', '.79', '.80', '.82')))
REPORTS = tuple(product(range(4), repeat=2))
HORIZON = 400


def matching(reports, rho):
    feasible = []
    for owners in product((-1, 0, 1), repeat=2):
        used = [owner for owner in owners if owner >= 0]
        if len(used) != len(set(used)):
            continue
        if any(owner >= 0 and not (0 < reports[owner] <= c and reports[owner] >= rho*c)
               for c, owner in zip(CAPACITY, owners)):
            continue
        feasible.append((sum(c for c, owner in zip(CAPACITY, owners) if owner >= 0), owners))
    best = max(capacity for capacity, _ in feasible)
    return tuple(owners for capacity, owners in feasible if capacity == best)


def packing(demands, values, available):
    best = None
    for placement in product((-1, 0, 1), repeat=2):
        if any(sum(demands[i] for i in range(2) if placement[i] == machine) > available[machine]
               for machine in range(2)):
            continue
        served = tuple(int(machine >= 0) for machine in placement)
        value = sum(v*s for v, s in zip(values, served))
        if best is None or value > best[0]:
            best = (value, served)
    return best


def expected_outcome(market, reports, rho, price=F(1), billing='legacy', quota=False, coalition=True):
    _, demands, values = market
    totals = {key: F(0) for key in ('revenue', 'utility', 'jobs', 'leased_core_hours', 'served_demand', 'value')}
    utilities = [F(0), F(0)]
    outcomes = matching(reports, rho)
    for owners in outcomes:
        payments = [F(0), F(0)]
        available = [0, 0]
        allocations = [0, 0]
        for j, (c, owner) in enumerate(zip(CAPACITY, owners)):
            if owner < 0:
                continue
            b = reports[owner]
            payments[owner] = price * (c if billing == 'capacity' else F(c+4*b, 5))
            available[j] = b if quota else c
            allocations[owner] = available[j]
        if coalition:
            value, served = packing(demands, values, available)
        else:
            served = tuple(int(allocated >= need) for allocated, need in zip(allocations, demands))
            value = sum(v*s for v, s in zip(values, served))
        revenue = sum(payments)
        result = {'revenue': revenue, 'utility': value-revenue, 'jobs': sum(served),
                  'leased_core_hours': sum(c for c, owner in zip(CAPACITY, owners) if owner >= 0),
                  'served_demand': sum(d*s for d, s in zip(demands, served)), 'value': value}
        for key in totals:
            totals[key] += F(result[key]) / len(outcomes)
        for i in range(2):
            utilities[i] += (values[i]*served[i]-payments[i]) / len(outcomes)
    totals['individual_utilities'] = tuple(utilities)
    return totals


def best_response(market, rho, price=F(1), billing='legacy', quota=False):
    options = [(reports, expected_outcome(market, reports, rho, price, billing, quota))
               for reports in REPORTS]
    best_utility = max(record['utility'] for _, record in options)
    optimal = [(reports, record) for reports, record in options if record['utility'] == best_utility]
    worst_revenue = min(record['revenue'] for _, record in optimal)
    reports, record = next((reports, record) for reports, record in optimal if record['revenue'] == worst_revenue)
    return {'selected_reports': reports, **record,
            'all_optimal_reports': [reports for reports, _ in optimal],
            'worst_revenue_tie_count': sum(record['revenue'] == worst_revenue for _, record in optimal)}


def pure_nash(market, rho, price, billing):
    outcomes = {reports: expected_outcome(market, reports, rho, price, billing, coalition=False)
                for reports in REPORTS}
    equilibria = []
    for reports, outcome in outcomes.items():
        if all(outcome['individual_utilities'][buyer] >= outcomes[tuple(
            action if index == buyer else reports[index] for index in range(2))]['individual_utilities'][buyer]
               for buyer in range(2) for action in range(4)):
            equilibria.append({'reports': reports, **outcome})
    return equilibria


def aggregate(records):
    return {key: sum(row[key] for row in records)/len(records) for key in
            ('revenue', 'utility', 'jobs', 'leased_core_hours', 'served_demand', 'value')}


def serializable(value):
    if isinstance(value, F):
        return {'exact': str(value), 'decimal': float(value)}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main():
    baseline = [best_response(market, F(1, 2)) for market in MARKETS]
    assert all(row['revenue'] == F('3.4') and row['jobs'] == 2 for row in baseline)
    grid = []
    invariance_checks = 0
    independent_rows = []
    for price in PRICE_GRID:
        rho_rows = []
        for rho in RHOS:
            market_rows = []
            for market in MARKETS:
                row = best_response(market, rho, price, 'capacity')
                market_rows.append({'market': market[0], **row})
                if price < F('.8'):
                    assert row['revenue'] == 5*price
                    assert row['jobs'] == 2
                    assert row['utility'] == sum(market[2])-5*price
                    full_options = [expected_outcome(market, report, rho, price, 'capacity')
                                    for report in REPORTS]
                    # Every non-full allocation is strictly below the attainable
                    # stage-optimal utility, including mixed matching outcomes.
                    assert all(o['utility'] < row['utility'] for o in full_options
                               if o['leased_core_hours'] < 5)
                    invariance_checks += 1
            summary = aggregate(market_rows)
            rho_rows.append({'rho': rho, 'markets': market_rows, 'mean_per_round': summary,
                             'mean_episode_revenue': HORIZON*summary['revenue'],
                             'mean_episode_jobs': HORIZON*summary['jobs'],
                             'relative_revenue_gain_vs_v1_fixed': summary['revenue']/F('3.4')-1})
        grid.append({'price': price, 'rho_results': rho_rows})
    chosen = F('.78')
    for rho in RHOS:
        for market in MARKETS:
            ne = pure_nash(market, rho, chosen, 'capacity')
            assert ne
            if rho in (F(1, 2), F(1)):
                assert all(row['revenue'] == 5*chosen and row['jobs'] == 2 for row in ne)
            independent_rows.append({'market': market[0], 'rho': rho, 'pure_nash_count': len(ne),
                                     'all_equilibria_full_service': all(row['jobs'] == 2 for row in ne),
                                     'equilibria': ne})
    decomposition = []
    for billing, price in product(('legacy', 'capacity'), (F(1), chosen)):
        rows = [best_response(market, F(1, 2), price, billing) for market in MARKETS]
        decomposition.append({'billing': billing, 'price': price,
                              'markets': [{'market': market[0], **row} for market, row in zip(MARKETS, rows)],
                              'mean_per_round': aggregate(rows)})
    quota = [best_response(market, F(1, 2), quota=True) for market in MARKETS]
    assert quota[1]['jobs'] == 0
    outside = []
    for market in (('E_packable', (1, 2), (F(2), F(5))),
                   ('B_value_minus10percent', (3, 2), (F('2.16'), F('1.44')))):
        mechanisms = []
        for billing, price in (('legacy', F(1)), ('capacity', F('.78')), ('capacity', F('.70'))):
            grand = best_response(market, F(1, 2), price, billing)
            independent = pure_nash(market, F(1, 2), price, billing)
            mechanisms.append({'billing': billing, 'price': price, 'grand': grand,
                               'independent_pure_equilibria': independent})
        outside.append({'market': market[0], 'demands': market[1], 'values': market[2],
                        'mechanisms': mechanisms})
    data = {
        'status': 'exact_fraction_enumeration_complete',
        'horizon': HORIZON, 'capacities': CAPACITY, 'registered_buyers': 2,
        'price_is_committed_for_the_episode': True,
        'coalition': 'All buyers can share leased capacity, pack indivisible jobs and freely transfer payments.',
        'matching': 'Maximum physical allocated capacity, uniform over all optimal matchings.',
        'tie_rule': 'Maximize expected joint utility, then minimize platform expected revenue, then lexicographic reports.',
        'baseline_v1_fixed_half': {'markets': baseline, 'mean_per_round': aggregate(baseline)},
        'capacity_price_grid': grid,
        'chosen_price_pure_nash_diagnostics': independent_rows,
        'independent_buyer_warning': 'In B at rho=0 or rho=2/3, an m=3 buyer can underreport while the m=2 buyer declines to enter: entering creates an expensive random VM assignment. These pure equilibria complete one job. At rho=1/2 and rho=1 all enumerated pure equilibria serve both jobs. The grand-coalition dynamic proof does not establish independent-agent convergence.',
        'billing_and_price_decomposition_at_rho_half': decomposition,
        'design_stage_out_of_catalogue_counterexamples': outside,
        'out_of_catalogue_warning': 'E fits both jobs on one VM, so legitimate sharing saves paid capacity; capacity billing does not eliminate this coalition saving. A 10% lower valuation in B makes p=.78 unaffordable, whereas p=.70 serves both. These are exact design-stage examples, not a preregistered statistical generalization test.',
        'hard_report_quota_original_tariff': {'markets': quota, 'mean_per_round': aggregate(quota),
             'B_market_budget_bound': {'reason': 'Payments >= sum enforced quota >= completed demand S; B value=.8*S. Individual rationality of the grand coalition requires subsidy >= .2*S.',
                 'episode_subsidy_budget': 50, 'maximum_completed_demand_core_hours': 250,
                 'full_completed_demand_core_hours': 2000, 'max_fraction': F(1, 8),
                 'subsidy_needed_for_95_percent_completed_demand': 380}},
        'dynamic_guarantee_in_this_catalogue': {
            'assumptions': ['A committed capacity unit price 0<p<.8', 'No rebates or switching/operating costs',
                'Only rho may depend on history; no change to capacities, workloads, price or feasible full allocation',
                'Buyers maximize their coalition total utility over the finite horizon',
                'The specified four stationary markets; all jobs are indivisible and feasible when both VMs are leased'],
            'proof': [
                'For every rho in [0,1], joint reports (3,2) admit the unique full matching; the coalition can rearrange its jobs afterward.',
                'In each of the four markets, full leasing gives U*=sum(values)-5p and strictly dominates every proper subset of leased VMs when 0<p<.8.',
                'U* does not depend on rho or history. Every strategy has horizon utility <= H*U*, and full leasing every round attains it under every platform threshold policy.',
                'No forward-looking sacrifice can improve utility above that attainable bound. Every best response leases both VMs in every round, so receipts=5pH and completed jobs=2H.',
                'At p=.8, market B becomes indifferent between participation and exit; the strict guarantee fails under platform-pessimistic tie-breaking.'
            ],
            'what_it_does_not_prove': ['Truthful individual resource reports', 'Breaking the coalition',
                'Revenue robustness for unobserved types or changed job packing possibilities',
                'A dynamic advantage over fixed thresholds with the same capacity tariff',
                'Convergence of bounded learners or a claim about real cloud demand'],
        },
        'checks': {'strict_full_leasing_market_rho_price_cases': invariance_checks,
                   'exact_joint_report_profiles_per_case': len(REPORTS),
                   'pure_nash_market_rho_cases_at_price_078': len(independent_rows)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'mechanism_audit.json').write_text(json.dumps(serializable(data), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': data['status'], 'strict_full_leasing_checks': invariance_checks,
        'price_078_episode_revenue': float(5*chosen*HORIZON),
        'baseline_episode_revenue': 1360, 'capacity_price1_mean_revenue': float(decomposition[2]['mean_per_round']['revenue']),
        'output': str(OUT/'mechanism_audit.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
