"""Compare this crate's Rust polars line with the one Python polars links against.

Python polars and the Rust `polars` crate version independently, and the plugin
bridge between them is checked at load time by the `polars_expr` macro. When
polars ships a Python release built on a new Rust crate line, the wheel published
here stops loading into it -- and nothing else in the repository notices, because
no other job installs a released polars and registers the plugin against it.

So this reads the `polars` entry out of our `Cargo.lock`, asks PyPI which Python
polars is current, reads that release's own `Cargo.lock` from the polars repo,
and compares the two lines. A mismatch means this crate has to be rebuilt and
re-released before that polars can use it.

Run by `.github/workflows/rebuild-trigger.yml` on a schedule.

Exit codes
    0   both lines agree
    1   polars moved Rust lines; a rebuild is needed
    2   either version could not be determined, so nothing was learned
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

PYPI_JSON = "https://pypi.org/pypi/polars/json"
POLARS_REPO_RAW = "https://raw.githubusercontent.com/pola-rs/polars"
USER_AGENT = "polars-ros rebuild trigger"
TIMEOUT = 30

STATUS_IN_SYNC = "in-sync"
STATUS_REBUILD = "rebuild-needed"
STATUS_ERROR = "error"


class LookupError(RuntimeError):
    """A version could not be determined; distinct from a real mismatch."""


def rust_line(version: str) -> str:
    """The compatibility line of a Rust crate version.

    Rust pre-1.0 crates are only compatible within `0.<minor>`, which is why
    `0.55.1` and `0.55.2` are interchangeable here but `0.54.4` is not.
    """
    parts = version.split(".")
    if not parts or not parts[0].isdigit():
        raise ValueError(f"unparsable version: {version!r}")
    if parts[0] == "0" and len(parts) > 1:
        return f"0.{parts[1]}"
    return parts[0]


def line_from_cargo_lock(lock_text: str, crate: str = "polars") -> str:
    """The Rust line for `crate` as recorded in a Cargo.lock body."""
    for block in lock_text.split("[[package]]"):
        if not re.search(rf'^name\s*=\s*"{re.escape(crate)}"\s*$', block, re.M):
            continue
        found = re.search(r'^version\s*=\s*"([^"]+)"', block, re.M)
        if found is None:
            raise LookupError(f"{crate} appears in the lockfile without a version")
        return rust_line(found.group(1))
    raise LookupError(f"no {crate} entry in the lockfile")


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def our_line(repo_root: str) -> str:
    try:
        with open(os.path.join(repo_root, "Cargo.lock"), encoding="utf-8") as fh:
            return line_from_cargo_lock(fh.read())
    except OSError as exc:
        raise LookupError(f"cannot read our Cargo.lock: {exc}") from exc


def their_line(polars_version: str) -> str:
    """The Rust line the given Python polars release links against.

    polars tags Python releases as `py-<version>`. A bare `<version>` tag is
    tried as a fallback in case that convention changes.
    """
    last_error: Exception | None = None
    for tag in (f"py-{polars_version}", polars_version):
        try:
            return line_from_cargo_lock(fetch(f"{POLARS_REPO_RAW}/{tag}/Cargo.lock"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
        except LookupError as exc:
            last_error = exc
    raise LookupError(
        f"could not read polars {polars_version} Cargo.lock: {last_error}"
    )


def latest_polars_version() -> str:
    try:
        payload = json.loads(fetch(PYPI_JSON))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
        raise LookupError(f"could not query PyPI for polars: {exc}") from exc
    version = payload.get("info", {}).get("version")
    if not version:
        raise LookupError("PyPI response had no info.version")
    return str(version)


def write_github_outputs(status: str, detail: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        # Single line by construction: the workflow interpolates this directly.
        fh.write(f"status={status}\ndetail={detail.replace(chr(10), ' ')}\n")


def write_step_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown.rstrip() + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--polars-version",
        help="inspect a specific polars release instead of the newest on PyPI",
    )
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="directory holding this project's Cargo.lock",
    )
    args = parser.parse_args(argv)

    polars_version = args.polars_version or ""
    try:
        polars_version = polars_version or latest_polars_version()
        ours = our_line(args.repo_root)
        theirs = their_line(polars_version)
    except LookupError as exc:
        detail = f"could not determine: {exc}"
        print(f"ERROR: {detail}")
        write_github_outputs(STATUS_ERROR, detail)
        write_step_summary(f"## polars Rust line check\n\n**Not checked:** {exc}")
        return 2

    if ours == theirs:
        detail = (
            f"in sync: python polars {polars_version} links Rust polars "
            f"{theirs}, we build against {ours}"
        )
        print(f"OK: {detail}")
        write_github_outputs(STATUS_IN_SYNC, detail)
        write_step_summary(
            "## polars Rust line check\n\n"
            f"In sync. Python polars `{polars_version}` links Rust polars "
            f"`{theirs}`; this crate builds against `{ours}`.\n"
        )
        return 0

    detail = (
        f"rebuild needed: python polars {polars_version} links Rust polars "
        f"{theirs} but we build against {ours}; the wheel will not load into "
        f"that polars until this crate is rebuilt and released"
    )
    print(f"MISMATCH: {detail}")
    write_github_outputs(STATUS_REBUILD, detail)
    write_step_summary(
        "## polars Rust line check\n\n"
        f"**Rebuild needed.** Python polars `{polars_version}` now links Rust "
        f"polars `{theirs}`, while this crate builds against `{ours}`.\n\n"
        "The plugin bridge checks this at load time, so the published wheel "
        "will fail to register against that polars.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
