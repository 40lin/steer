"""Exact economic sanity checks, NOT a trained steering-policy experiment.

Python standard library only. All payments and utilities use Fraction.
Time slot = one hour. VM capacity fixed in slot. One whole VM per account.
Report 0 is an outside/opt-out action; positive reports can exceed true demand.
Match maximum TOTAL CAPACITY, then randomize uniformly across tied matchings.
Coalition pools VM access; each indivisible job fits on exactly one VM, with
colocation permitted if remaining capacity allows. Internal transfers cancel.
"""
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
from pathlib import Path
import json

ALPHA = F(1, 5)
RHO = (F(0), F(1, 2), F(2, 3), F(4, 5), F(1))


def money(capacity, report, rebate=F(0)):
    return ALPHA * capacity + (1 - ALPHA - rebate) * report


@lru_cache(None)
def matchings(capacities, reports, rho):
    n = len(reports)
    candidates = []
    for owners in product(range(-1, n), repeat=len(capacities)):
        actual = [i for i in owners if i >= 0]
        if len(actual) != len(set(actual)):
            continue
        if any(i >= 0 and not (0 < reports[i] <= c and reports[i] >= rho * c)
               for c, i in zip(capacities, owners)):
            continue
        q = sum(c for c, i in zip(capacities, owners) if i >= 0)
        candidates.append((q, owners))
    qmax = max(q for q, _ in candidates)
    return tuple(owners for q, owners in candidates if q == qmax)


def optimal_matching_count_dp(capacities, reports, rho):
    """Independent value/count DP underlying exact uniform optimal sampling."""
    @lru_cache(None)
    def solve(j, used):
        if j == len(capacities):
            return (0, 1)
        options = [solve(j + 1, used)]
        c = capacities[j]
        for i, b in enumerate(reports):
            if not (used & (1 << i)) and 0 < b <= c and b >= rho * c:
                value, count = solve(j + 1, used | (1 << i))
                options.append((c + value, count))
        best = max(value for value, _ in options)
        return best, sum(count for value, count in options if value == best)
    return solve(0, 0)


def packing_value(demands, values, capacities):
    """Exact job packing. Unserved jobs yield zero. No cross-VM job splitting."""
    @lru_cache(None)
    def solve(i, remaining):
        if i == len(demands):
            return F(0)
        best = solve(i + 1, remaining)
        for j, c in enumerate(remaining):
            if c >= demands[i]:
                new = list(remaining)
                new[j] -= demands[i]
                best = max(best, values[i] + solve(i + 1, tuple(new)))
        return best
    return solve(0, tuple(capacities))


def evaluate(demands, values, capacities, reports, rho, rebate=F(0)):
    ms = matchings(capacities, reports, rho)
    private = [F(0)] * len(demands)
    grand = F(0)
    revenue = F(0)
    volume = F(0)
    completed_value = F(0)
    for owners in ms:
        leased = []
        total_payment = F(0)
        for c, i in zip(capacities, owners):
            if i < 0:
                continue
            assert reports[i] <= c and reports[i] >= rho * c
            p = money(c, reports[i], rebate)
            total_payment += p
            private[i] += (values[i] if c >= demands[i] else F(0)) - p
            leased.append(c)
        value = packing_value(demands, values, leased)
        completed_value += value
        grand += value - total_payment
        revenue += total_payment
        volume += sum(leased)
    k = len(ms)
    return {
        'private_utilities': tuple(x / k for x in private),
        'coalition_utility': grand / k,
        'revenue': revenue / k,
        'volume': volume / k,
        'coalition_completed_value': completed_value / k,
    }


def solve_case(name, demands, values, capacities):
    n = len(demands)
    actions = range(max(capacities) + 1)
    profiles = tuple(product(actions, repeat=n))
    result = {'name': name, 'demands': demands, 'values': values,
              'capacities': capacities, 'results': []}
    floor_checks = 0
    matching_count_checks = 0
    for rho in RHO:
        ev = {b: evaluate(demands, values, capacities, b, rho) for b in profiles}
        for b in profiles:
            ms = matchings(capacities, b, rho)
            value, count = optimal_matching_count_dp(capacities, b, rho)
            assert count == len(ms)
            assert value == sum(c for c, i in zip(capacities, ms[0]) if i >= 0)
            matching_count_checks += 1
        ne = []
        for b in profiles:
            u = ev[b]['private_utilities']
            if all(ev[b[:i] + (a,) + b[i+1:]]['private_utilities'][i] <= u[i]
                   for i in range(n) for a in actions):
                ne.append(b)
        best_u = max(x['coalition_utility'] for x in ev.values())
        best = [b for b in profiles if ev[b]['coalition_utility'] == best_u]
        revenues = [ev[b]['revenue'] for b in best]
        worst = min(best, key=lambda b: (ev[b]['revenue'], b))
        truth = ev[tuple(demands)]
        row = {
            'rho': rho,
            'truth_revenue': truth['revenue'],
            'pure_noncoalition_NE': [
                {'reports': b, 'revenue': ev[b]['revenue'],
                 'individual_utilities': ev[b]['private_utilities'],
                 'coalition_gain_from_NC': max(F(0), best_u - sum(ev[b]['private_utilities']))}
                for b in ne],
            'coalition_optimal_utility': best_u,
            'coalition_optimal_reports': best,
            'coalition_revenue_range': [min(revenues), max(revenues)],
            'worst_seller_optimum': worst,
            'coalition_volume': ev[worst]['volume'],
        }
        result['results'].append(row)
        for rebate in (F(0), F(1, 10)):
            assert rebate <= 1 - ALPHA
            gamma = ALPHA + (1 - ALPHA - rebate) * rho
            for b in profiles:
                for owners in matchings(capacities, b, rho):
                    q = sum(c for c, i in zip(capacities, owners) if i >= 0)
                    paid = sum((money(c, b[i], rebate) for c, i in zip(capacities, owners) if i >= 0), F(0))
                    assert paid >= gamma * q
                    floor_checks += 1
    result['floor_checks'] = floor_checks
    result['matching_count_checks'] = matching_count_checks
    return result


def serialize(x):
    if isinstance(x, F):
        return {'exact': str(x), 'decimal': float(x)}
    if isinstance(x, dict):
        return {k: serialize(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serialize(v) for v in x]
    return x


def main():
    cases = [
        solve_case('high_value_two_buyers', (3, 2), (F(9), F(5)), (3, 2)),
        solve_case('low_value_two_buyers', (3, 2), (F(12, 5), F(8, 5)), (3, 2)),
        solve_case('overreporting_counterexample', (2, 2), (F(5), F(5)), (3, 2)),
        solve_case('three_buyers', (3, 2, 2), (F(6), F(4), F(4)), (3, 2)),
    ]
    high = cases[0]['results']
    assert high[0]['coalition_revenue_range'] == [F(13, 5), F(13, 5)]
    assert high[2]['coalition_revenue_range'] == [F(21, 5), F(21, 5)]
    assert high[-1]['coalition_revenue_range'] == [F(5), F(5)]
    assert cases[1]['results'][-1]['coalition_volume'] == 0
    over_ne = cases[2]['results'][-1]['pure_noncoalition_NE']
    assert any(any(b > m for b, m in zip(r['reports'], (2, 2))) for r in over_ne)
    # Budget reservation proof check on a finite sample: advertised rebate bound
    # kappa*p*available_capacity <= remaining budget; actual <= advertised bound.
    budget_checks = 0
    for rem in (F(0), F(1, 10), F(1), F(10)):
        kappa = min(F(1, 10), rem / 5)
        for b in product(range(4), repeat=2):
            for owners in matchings((3, 2), b, F(2, 3)):
                spent = sum((kappa * b[i] for i in owners if i >= 0), F(0))
                assert 0 <= spent <= rem
                budget_checks += 1
    data = {
        'status': 'exact_economic_sanity_check_only',
        'assumptions': {'price': 1, 'alpha': ALPHA, 'slot_hours': 1,
                        'operating_cost': 0, 'optout_report': 0,
                        'matching': 'maximum_total_capacity_uniform_ties',
                        'report_range': '0..max(capacity), includes overreporting',
                        'sharing': 'indivisible_job_packing_no_cross_VM_splitting',
                        'transfers': 'unrestricted_budget_balanced_within_grand_coalition'},
        'cases': cases,
        'assertions_passed': True,
        'floor_checks': sum(c['floor_checks'] for c in cases),
        'budget_checks': budget_checks,
        'matching_count_checks': sum(c['matching_count_checks'] for c in cases),
        'limitations': ['No learned steering policy was trained.',
                        'Stage-game checks do not certify a repeated dynamic mechanism.',
                        'Truthful reports are a counterfactual, not generally an equilibrium.',
                        'Operating costs and rebates omitted from reported game comparisons; floor separately tests rebates.'],
    }
    out = Path(__file__).resolve().parents[1] / 'results'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'micro_validation.json').write_text(json.dumps(serialize(data), ensure_ascii=False, indent=2), encoding='utf-8')
    for case in cases:
        print(case['name'])
        for row in case['results']:
            nes = [(r['reports'], str(r['revenue'])) for r in row['pure_noncoalition_NE']]
            print(' rho=',str(row['rho']),'NC_NE=',nes,'C_R=',[str(x) for x in row['coalition_revenue_range']],
                  'C_Q=',str(row['coalition_volume']),'C_reports=',row['worst_seller_optimum'])
    print('PASS:',data['floor_checks'],'floor checks;',budget_checks,'budget checks;',
          data['matching_count_checks'],'matching count checks; exact assertions passed.')


if __name__ == '__main__':
    main()
