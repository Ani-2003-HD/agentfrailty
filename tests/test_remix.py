"""
Remix tests -- the ablation gate.

The ablation is only valid if a variant differs from its base in EXACTLY the
intended way. If relabelling also perturbed the graph, or revaluing also moved
the ids, the two factors would be confounded and the answer meaningless.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentfrailty.envs.ledger import make_task, remix_task, solve  # noqa: E402

FAILURES = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


BASE = make_task(seed=3, n=6, n_distractors=12)


def shape(t):
    """Wiring only: positions, not labels. Two isomorphic graphs share this."""
    idx = {rid: i for i, rid in enumerate(t.records)}
    return [(idx[r.id], idx[r.next] if r.next is not None else None)
            for r in t.records.values()]


def test_identity_remix():
    c = remix_task(BASE)
    check("no seeds -> same ids", list(c.records) == list(BASE.records))
    check("no seeds -> same values",
          [r.value for r in c.records.values()]
          == [r.value for r in BASE.records.values()])
    check("no seeds -> same goal", c.goal_total == BASE.goal_total)


def test_values_only():
    v = remix_task(BASE, values_seed=99)
    check("values-only: ids unchanged", v.canonical_path == BASE.canonical_path)
    check("values-only: start unchanged", v.start_id == BASE.start_id)
    check("values-only: wiring unchanged", shape(v) == shape(BASE))
    check("values-only: values DID change",
          [r.value for r in v.records.values()]
          != [r.value for r in BASE.records.values()])
    check("values-only: goal recomputed",
          v.goal_total == sum(v.records[x].value for x in v.canonical_path))


def test_ids_only():
    i = remix_task(BASE, ids_seed=99)
    check("ids-only: ids DID change", i.canonical_path != BASE.canonical_path)
    check("ids-only: no id survives",
          not (set(i.records) & set(BASE.records)))
    check("ids-only: wiring unchanged (isomorphic)", shape(i) == shape(BASE))
    check("ids-only: values unchanged in position",
          [r.value for r in i.records.values()]
          == [r.value for r in BASE.records.values()])
    check("ids-only: goal identical", i.goal_total == BASE.goal_total)


def test_both():
    b = remix_task(BASE, ids_seed=5, values_seed=5)
    check("both: ids changed", b.canonical_path != BASE.canonical_path)
    check("both: values changed",
          [r.value for r in b.records.values()]
          != [r.value for r in BASE.records.values()])
    check("both: wiring still preserved", shape(b) == shape(BASE))


def test_variants_remain_solvable():
    ok = True
    for i_s in (None, 1, 2):
        for v_s in (None, 1, 2):
            t = remix_task(BASE, ids_seed=i_s, values_seed=v_s)
            path, total = solve(t)
            ok &= (len(path) == BASE.n and total == t.goal_total
                   and path == t.canonical_path)
    check("every variant solves in exactly n steps", ok)


def test_determinism():
    a = remix_task(BASE, ids_seed=7, values_seed=8)
    b = remix_task(BASE, ids_seed=7, values_seed=8)
    check("same seeds reproduce the variant", a.to_dict() == b.to_dict())
    c = remix_task(BASE, ids_seed=7, values_seed=9)
    check("different value seed differs", a.to_dict() != c.to_dict())


def test_no_shortcut_survives_remix():
    """The no-shortcut property must hold in variants too."""
    ok = True
    for i_s in (1, 2, 3):
        t = remix_task(BASE, ids_seed=i_s)
        for i in range(len(t.canonical_path) - 1):
            nxt = t.canonical_path[i + 1]
            revealers = [r for r, rec in t.records.items() if rec.next == nxt]
            ok &= (revealers == [t.canonical_path[i]])
    check("each path id still revealed by exactly one predecessor", ok)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} remix test groups\n")
    for fn in fns:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("all checks passed")
