# vendored from sfxnz/forge kit @ 3d294ff (via Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark kit/render.py @ 4eef42d,
# sha256 f2aa3151e21687179142262545db93e142456cf83e74821fa32e84429a6466e1).
# Changed here: a measured row with no evidence renders as "not yet measured on this pair"; a row that has a number
# but no evidence, an evidence path that does not exist, or evidence without all three numbers is an error (no file,
# no number); a line inside the run.sh block that is not a comment, SNAPSHOT=, SNAPSHOT_IN_CONTAINER= or
# NAME="${NAME:-…}" is an error.
# Regenerate the marked blocks of run.sh and README.md from recipe.yaml.
#
#   python3 kit/render.py            rewrite the generated blocks in place
#   python3 kit/render.py --check    exit 1 with a unified diff if any block is stale
#   python3 kit/render.py --strict   also exit 1 when a measured row has no evidence yet
#
# recipe.yaml is the source of truth. Nothing outside these markers is touched:
#   run.sh     # BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
#              # END generated
#   README.md  <!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
#              <!-- END generated defaults -->
#              <!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
#              <!-- END generated measured -->
#
# The measured block is a `Conditions:` line, the decode table, and a numbered note for each row
# that carries an optional `note`.
#
# Inside the run.sh block every NAME="${NAME:-value}" line takes its value from serve.env; blank and comment
# lines are kept verbatim, and SNAPSHOT= and SNAPSHOT_IN_CONTAINER= are derived from model.id. Any other line is refused. serve.env must list exactly those names, in run.sh order. README
# defaults rows may use {NAME} placeholders for serve.env values. Needs python3 and PyYAML.
import argparse
import difflib
import re
import sys
from pathlib import Path

import yaml

RUN_BEGIN = "# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py"
RUN_END = "# END generated"
DEFAULT_LINE = re.compile(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-.*\}"$')
PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
DEFAULTS_HEADER = ["| Setting | Value |", "|---|---|"]
MEASURED_HEADER = [
    "| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |",
    "|---|---|---:|---:|---:|",
]
NO_EVIDENCE = ("", "null", "~")  # BaseLoader keeps `null` as the string "null"
NOT_MEASURED = "not yet measured on this pair"


def md_markers(name):
    return (
        f"<!-- BEGIN generated {name} from recipe.yaml — edit recipe.yaml and run kit/render.py -->",
        f"<!-- END generated {name} -->",
    )


def find_block(lines, begin, end, path):
    """Return (start, stop) so that lines[start:stop] is the body between the marker lines."""
    starts = [i for i, line in enumerate(lines) if line == begin]
    if len(starts) != 1:
        sys.exit(f"render: {path}: need exactly one `{begin}` line, found {len(starts)}")
    try:
        stop = lines.index(end, starts[0] + 1)
    except ValueError:
        sys.exit(f"render: {path}: `{begin}` has no `{end}`")
    return starts[0] + 1, stop


def hub_dir(model_id):
    return "models--" + model_id.replace("/", "--")


def render_run_sh(text, env, model_id):
    lines = text.split("\n")
    start, stop = find_block(lines, RUN_BEGIN, RUN_END, "run.sh")
    body, seen = [], []
    hub = hub_dir(model_id)
    for line in lines[start:stop]:
        if line.startswith("SNAPSHOT="):
            body.append(f'SNAPSHOT="${{HF_CACHE}}/hub/{hub}/snapshots/${{SNAPSHOT_SHA}}"')
            continue
        if line.startswith("SNAPSHOT_IN_CONTAINER="):
            body.append(
                f'SNAPSHOT_IN_CONTAINER="${{HF_HOME_IN_CONTAINER}}/hub/{hub}/snapshots/${{SNAPSHOT_SHA}}"'
            )
            continue
        m = DEFAULT_LINE.match(line)
        if not m:
            if line.strip() and not line.lstrip().startswith("#"):
                sys.exit(f"render: run.sh: `{line}` inside the generated block is not NAME=\"${{NAME:-…}}\"; "
                         "put the value in recipe.yaml serve.env")
            body.append(line)
            continue
        name = m.group(1)
        if name not in env:
            sys.exit(f"render: run.sh sets {name} inside the generated block but serve.env has no {name}")
        seen.append(name)
        body.append(f'{name}="${{{name}:-{env[name]}}}"')
    if seen != list(env):
        sys.exit(
            "render: serve.env must list the generated run.sh defaults in run.sh order\n"
            f"  run.sh:      {' '.join(seen)}\n  recipe.yaml: {' '.join(env)}"
        )
    return "\n".join(lines[:start] + body + lines[stop:])


def fill(template, env):
    def value(m):
        name = m.group(1)
        if name not in env:
            sys.exit(f"render: README row uses {{{name}}} but serve.env has no {name}")
        return env[name]

    return PLACEHOLDER.sub(value, template)


def render_readme(text, recipe, repo):
    env = recipe["serve"]["env"]
    lines = text.split("\n")
    start, stop = find_block(lines, *md_markers("defaults"), "README.md")
    rows = [f"| {fill(k, env)} | {fill(v, env)} |" for row in recipe["readme"]["defaults"] for k, v in row.items()]
    lines[start:stop] = DEFAULTS_HEADER + rows
    start, stop = find_block(lines, *md_markers("measured"), "README.md")
    decode = recipe["measured"]["decode"]
    rows, notes = [], []
    for r in decode["rows"]:
        phase = r["phase"]
        if r.get("evidence", "") in NO_EVIDENCE:
            if any(r.get(k, "") not in NO_EVIDENCE for k in ("decode", "aggregate", "ttft_p50")):
                sys.exit(f"render: measured.decode {phase} c={r['concurrency']} has a number but no evidence file")
            rows.append(f"| {phase} | {r['concurrency']} | {NOT_MEASURED} | – | – |")
            continue
        label = f"measured.decode {phase} c={r['concurrency']}"
        if not (repo / r["evidence"]).is_file():
            sys.exit(f"render: {label}: evidence file {r['evidence']} does not exist")
        if any(r.get(k, "") in NO_EVIDENCE for k in ("decode", "aggregate", "ttft_p50")):
            sys.exit(f"render: {label} has an evidence file but not all of decode, aggregate and ttft_p50")
        if r.get("note"):  # optional per-row caveat, rendered as a numbered note under the table
            notes.append(r["note"])
            phase = f"{phase} (note {len(notes)})"
        rows.append(f"| {phase} | {r['concurrency']} | {r['decode']} | {r['aggregate']} | {r['ttft_p50']} s |")
    notes = [""] + [f"{i}. {n}" for i, n in enumerate(notes, 1)] if notes else []
    lines[start:stop] = [f"Conditions: {decode['conditions']}.", ""] + MEASURED_HEADER + rows + notes
    return "\n".join(lines)


def evidence_gaps(recipe):
    """Rows with no evidence yet (render_readme already refused a row whose evidence file is missing)."""
    return [f"measured.decode {r['phase']} c={r['concurrency']}" for r in recipe["measured"]["decode"]["rows"]
            if r.get("evidence", "") in NO_EVIDENCE]


def main():
    ap = argparse.ArgumentParser(description="Regenerate run.sh and README.md blocks from recipe.yaml.")
    ap.add_argument("--check", action="store_true", help="print a diff and exit 1 instead of writing")
    ap.add_argument("--strict", action="store_true", help="a row with no evidence yet is an error")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    recipe = yaml.load((repo / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)
    renderers = {
        "run.sh": lambda text: render_run_sh(text, recipe["serve"]["env"], recipe["model"]["id"]),
        "README.md": lambda text: render_readme(text, recipe, repo),
    }
    stale = False
    for name, render in renderers.items():
        old = (repo / name).read_text()
        new = render(old)
        if new == old:
            continue
        if args.check:
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))
            stale = True
        else:
            (repo / name).write_text(new)
            print(f"render: wrote {name}")
    none = evidence_gaps(recipe)
    for label in none:
        print(f"render: no evidence yet: {label}")
    if stale:
        sys.exit("render: generated blocks are stale. Run: python3 kit/render.py")
    if args.strict and none:
        sys.exit("render: --strict: every measured row needs an evidence file")


if __name__ == "__main__":
    main()
