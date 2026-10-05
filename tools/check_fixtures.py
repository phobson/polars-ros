"""Verify tests/conftest.py against the installed wqio's test_ros.py.

Parses wqio's test file with `ast` so the transcribed fixtures can be diffed
against the real source. Run it after editing `tests/conftest.py`::

    python tools/check_fixtures.py [--require]

Needs polars (to read tests/conftest.py) and the installed wqio, but not
numpy/pandas/scipy/probscale -- wqio is located on disk and parsed as text, never
imported (see `wqio_location`).

With wqio unavailable this exits 0 with a skip notice so it can run in a bare
environment. Pass --require (or set POLARS_ROS_REQUIRE_WQIO=1, which CI does) to
make that a failure instead.
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

from wqio_gate import add_require_flag, gate, required_from_env  # noqa: E402
from wqio_location import WqioNotInstalled, wqio_source  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
add_require_flag(_ap)
_args = _ap.parse_args()

# Checked before importing conftest, which needs polars: a bare environment
# should skip cleanly rather than die with ImportError.
try:
    _wqio_test_ros = wqio_source("tests", "test_ros.py")
except WqioNotInstalled as exc:
    raise SystemExit(
        gate(
            bool(_args.require) or required_from_env(),
            str(exc),
            "run: pip install -e '.[dev]'",
        )
    )

import conftest  # noqa: E402

TARGETS = {
    "HelselAppendixB": "helsel_appendix_b",
    "HelselArsenic": "helsel_arsenic",
    "RNADAdata": "rnada",
    "NoOp_ZeroND": "no_op_zero_nd",
    "OneND": "one_nd",
    "HalfDLs_80pctNDs": "half_dls_80pct_nds",
    "HaflDLs_OneUncensored": "hafl_dls_one_uncensored",
    "MaxCen_GT_MaxUncen": "max_cen_gt_max_uncen",
    "OnlyDL_GT_MaxUncen": "only_dl_gt_max_uncen",
}

NAN = float("nan")
INF = float("inf")

failures = []
skipped = []
SCOPE = {}


class Unknown:
    """A value that could not be evaluated statically."""

    def __init__(self, reason=""):
        self.reason = reason

    def __str__(self):
        return self.reason

    def __repr__(self):
        return f"Unknown({self.reason!r})"


def literal(node, scope=None):
    """Evaluate the numeric shapes wqio's fixtures use."""
    scope = scope if scope is not None else SCOPE
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "numpy" and node.attr in {"inf", "nan"}:
            return INF if node.attr == "inf" else NAN
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func in {"numpy.array", "numpy.asarray", "list"}:
            return literal(node.args[0], scope)
        if func in {"StringIO", "io.StringIO", "dedent", "textwrap.dedent"}:
            return literal(node.args[0], scope)
        if func == "numpy.random.lognormal":
            return Unknown(f"numpy.random.lognormal (needs numpy)")
    if isinstance(node, ast.BinOp):
        left = literal(node.left, scope)
        right = literal(node.right, scope)
        if isinstance(node.op, ast.Mult):
            if isinstance(right, int) and isinstance(left, list):
                return left * right
        if isinstance(node.op, ast.Add):
            if isinstance(left, list) and isinstance(right, list):
                return left + right
        return Unknown(ast.unparse(node)[:50])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = literal(node.operand, scope)
        if isinstance(inner, (int, float)):
            return -inner if isinstance(node.op, ast.USub) else inner
        return Unknown(ast.unparse(node)[:50])
    if isinstance(node, ast.List):
        return [literal(e, scope) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k, scope): literal(v, scope) for k, v in zip(node.keys, node.values, strict=False)}
    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        return Unknown(f"name {node.id}")
    return Unknown(ast.unparse(node)[:50])


def unwrap_str(node):
    while isinstance(node, ast.Call):
        node = node.args[0]
    return node.value


def dedent_rows(node):
    rows = []
    for line in unwrap_str(node).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] != "res":
            rows.append((float(parts[0]), parts[1] == "True"))
    return rows


def close(a, b, tol):
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return True
    return abs(a - b) <= tol


def check_list(name, want, got, tol=5e-6):
    if isinstance(want, Unknown):
        skipped.append(f"{name}: {want}")
        return
    if isinstance(got, Unknown):
        failures.append(f"{name}: conftest side not evaluable")
        return
    if len(want) != len(got):
        failures.append(f"{name}: length {len(got)} != wqio {len(want)}")
        return
    for i, (w, g) in enumerate(zip(want, got)):
        if not close(w, g, tol):
            failures.append(f"{name}[{i}]: {g!r} != wqio {w!r}")
            return


tree = ast.parse(_wqio_test_ros)
# Collect class attributes, then resolve inheritance.
own = {}
bases = {}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in TARGETS:
        scope = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                v = literal(stmt.value, scope)
                if isinstance(v, (int, float)):
                    scope[stmt.targets[0].id] = v

        attrs = {}
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)):
                continue
            name = stmt.targets[0].id
            if name not in {"res", "cen", "values", "decimal", "datastring", "cohn"}:
                continue
            if name == "datastring":
                attrs[name] = dedent_rows(stmt.value)
            elif name == "cohn":
                d = stmt.value.args[0]
                attrs[name] = {
                    literal(k, scope): literal(v, scope)
                    for k, v in zip(d.keys, d.values, strict=False)
                }
            else:
                attrs[name] = literal(stmt.value, scope)
        own[node.name] = attrs
        bases[node.name] = [b.id for b in node.bases]


def resolve(cls, attr, seen=None):
    seen = seen or set()
    if cls in seen:
        return Unknown("inheritance cycle")
    seen.add(cls)
    if attr in own.get(cls, {}):
        return own[cls][attr]
    for b in bases.get(cls, []):
        if b in own:
            got = resolve(b, attr, seen)
            if not isinstance(got, Unknown):
                return got
    return Unknown(f"{cls}.{attr} not found")


for wqio_name, key in TARGETS.items():
    case = conftest.LITERATURE_CASES[key]

    if wqio_name == "RNADAdata":
        rows = resolve(wqio_name, "datastring")
        check_list(f"{key}.res", [r for r, _ in rows], case.res)
        check_list(f"{key}.cen", [c for _, c in rows], case.cen)
    else:
        check_list(f"{key}.res", resolve(wqio_name, "res"), case.res)
        check_list(f"{key}.cen", resolve(wqio_name, "cen"), case.cen)

    check_list(f"{key}.values", resolve(wqio_name, "values"), case.values)

    want_dec = resolve(wqio_name, "decimal")
    if not isinstance(want_dec, Unknown) and want_dec != case.decimal:
        failures.append(f"{key}.decimal: {case.decimal} != wqio {want_dec}")

    want_cohn = resolve(wqio_name, "cohn")
    if isinstance(want_cohn, Unknown):
        failures.append(f"{key}: wqio cohn not resolvable")
        continue
    if not case.cohn:
        # Deliberate: for the two "censored above max uncensored" cases the
        # extra censored rows add detection limits, so wqio's inherited Cohn
        # expectations no longer apply and we assert nothing.
        skipped.append(f"{key}.cohn: intentionally not asserted")
        continue
    for col, want in want_cohn.items():
        got = case.cohn.get(col)
        if isinstance(want, Unknown):
            failures.append(f"{key}.cohn.{col}: wqio side {want}")
            continue
        if not want:
            if got:
                failures.append(f"{key}.cohn.{col}: expected empty, got {got}")
            continue
        if got is None:
            failures.append(f"{key}.cohn: missing column {col}")
            continue
        # wqio appends one padding row; for prob_exceedance it is 0.0 rather
        # than NaN, so trim a trailing entry whenever wqio has exactly one more.
        if len(want) == len(got) + 1:
            want = want[:-1]
        check_list(f"{key}.cohn.{col}", want, got, tol=5e-5)

# wqio drops censored rows strictly above the max uncensored result.
for key, case in conftest.LITERATURE_CASES.items():
    if len(case.res) != len(case.cen):
        failures.append(f"{key}: len(res)={len(case.res)} != len(cen)={len(case.cen)}")
        continue
    uncensored = [v for v, c in zip(case.res, case.cen, strict=True) if not c]
    if not uncensored:
        continue
    top = max(uncensored)
    kept = sum(1 for v, c in zip(case.res, case.cen, strict=True) if (not c) or v <= top)
    if len(case.values) != kept:
        failures.append(f"{key}: {kept} rows survive the drop rule, values has {len(case.values)}")

# basic_data / ros_sorted_data must agree with wqio's expected_sorted fixture.
sorted_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "expected_sorted":
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call) and ast.unparse(stmt.func).endswith("DataFrame"):
                rows = literal(stmt.args[0])
                sorted_src = [(r["conc"], r["censored"]) for r in rows]
if sorted_src is None:
    failures.append("could not parse wqio's expected_sorted fixture")
else:
    base_pairs = list(zip(conftest.BASE_RESULTS, conftest.BASE_CENSORED, strict=True))
    order = sorted(range(len(base_pairs)), key=lambda i: (not base_pairs[i][1], base_pairs[i][0]))
    mine = [base_pairs[i] for i in order]
    if mine != sorted_src:
        failures.append("ros_sorted_data order/content does not match wqio's expected_sorted")
    else:
        print(f"ros_sorted_data matches wqio's expected_sorted ({len(mine)} rows)")

# basic_data must be the original CSV order.
csv_rows = None
h = ast.parse(wqio_source("tests", "helpers.py"))
for n in ast.walk(h):
    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("res,qual"):
        csv_rows = [
            (float(a), b.strip() == "ND")
            for a, b in (line.split(",") for line in n.value.strip().splitlines()[1:])
        ]
if csv_rows != base_pairs:
    failures.append("basic_data does not match getTestROSData's CSV order")
else:
    print(f"basic_data matches getTestROSData's CSV ({len(base_pairs)} rows)")

print()
if skipped:
    print("skipped:")
    for s in sorted(set(skipped)):
        print(f"  - {s}")
    print()

if failures:
    print(f"FAIL ({len(failures)}):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("PASS: conftest fixtures match wqio's test file")