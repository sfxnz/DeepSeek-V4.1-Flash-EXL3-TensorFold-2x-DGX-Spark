#!/usr/bin/env python3
"""Static and isolated checks of run.sh / stop.sh / the gate tool. No Docker, no GPU, stdlib only
(PyYAML for the render checks, skipped without it).

    python3 -m unittest discover -s tests -q
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - CI installs it
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
# bench_decode.py is byte-identical to the vLLM sibling's frozen ruler
# (sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de9146, bench_decode.py).
BENCH_DECODE_SHA256 = "3172cbc4558ae9c18a2a5844b824129c0cd9ab4d0371c148c1c31a0999708813"
# Files copied unchanged from elsewhere: the sha256 of the source file.
VERBATIM = {
    "bench_decode.py": BENCH_DECODE_SHA256,
    "smoke_chat.py": "e1516eb218e5b5e16aeaf65eea2fe7dd195bdbe4dc62d403cd3cb6a712685e96",
    "tests/test_bench_decode.py": "bab0efee6373df8f14a77d3aa50ecdc9a69fb07786f59d8559336ea01fe020e8",
    "tests/test_smoke_chat.py": "7dbd9d977db194c349e7b12992d986cbf758c203224412c5bc52334ed367d35f",
}
TF_REPO = "https://github.com/sfxnz/TensorFold.git"
TF_SHA = "903a1e8af62c8f46eceee6b95481706ada30ae49"
SNAPSHOT_SHA = "982b70452f399814f56b46272fd30394ae10d58c"
CONFIG_SHA256 = "6469adab394edead3eec148323e7471582d08acdf60a438bf0c9e815b69f36c5"
INDEX_SHA256 = "91731e4af38696bd4c09e960f4b599d1d49f35d445e4f88a43cc355d9e139f03"
BASE_DIGEST = "sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash"
SNAP_IN_CONTAINER = f"/cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/{SNAPSHOT_SHA}"
RECEIPT_ENV = "evidence/s0-engine-receipts/environment.txt"
# Receipt flags run.sh sets from its own variables: the rank, the ports, the bind host and the window (CONTEXT).
PER_BOOT_FLAGS = {"--rank", "--master-port", "--host", "--port", "--context"}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _defaults() -> dict[str, str]:
    """run.sh's generated defaults (rendered from recipe.yaml; render --check keeps them equal)."""
    block = _read("run.sh").split("# BEGIN generated", 1)[1].split("# END generated", 1)[0]
    return dict(re.findall(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-(.*)\}"$', block, re.M))


def _receipt_serve_argv() -> list[str]:
    """Both receipt ranks' `serve` argv from environment.txt, rank 0's extras left out."""
    line = next(l for l in _read(RECEIPT_ENV).splitlines() if l.startswith("serve (both ranks"))
    return ["tensorfold", "serve"] + shlex.split(line.split(" serve ", 1)[1].split(" [rank 0:", 1)[0])


def _generated_names() -> list[str]:
    block = _read("run.sh").split("# BEGIN generated", 1)[1].split("# END generated", 1)[0]
    return re.findall(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-', block, re.M)


def _clean_env(**extra: str) -> dict[str, str]:
    """The caller's env without any variable run.sh reads, plus `extra`."""
    drop = set(_generated_names()) | {"ROLE", "VALIDATE_ONLY", "HF_HUB_DISABLE_XET", "FORCE_UNSAFE_MEM_GATE",
                                      "IMAGE_ONLY"}
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("TF_DSV41_")}
    env.update(extra)
    return env


def _run_sh(validate: str = "1", script: Path | None = None, **extra: str) -> subprocess.CompletedProcess[str]:
    env = _clean_env(**extra)
    env["VALIDATE_ONLY"] = validate
    path = script or ROOT / "run.sh"
    return subprocess.run(["bash", str(path)], check=False, capture_output=True, text=True, cwd=str(path.parent), env=env)


def _func_src(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", src, re.M | re.S)
    if m is None:
        m = re.search(rf"^{name}\(\) \{{[^\n]*\}}\n", src, re.M)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(0)


def _func_body(src: str, name: str) -> str:
    return _func_src(src, name).split("{", 1)[1]


def _stubs(tmp: Path, **scripts: str) -> dict[str, str]:
    """Executable stubs on PATH; returns an env with them first."""
    for name, body in scripts.items():
        p = tmp / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")


def _bash(script: str, env: dict[str, str] | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=timeout, check=False)


def _shard(path: Path, nbytes: int, truncate: int = 0) -> int:
    header = json.dumps({"w": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (nbytes - truncate))
    return path.stat().st_size


class GuardTests(unittest.TestCase):
    """Each refusal fires with its reason before VALIDATE_ONLY exits."""

    def refused(self, needle: str, **env: str) -> None:
        proc = _run_sh(**env)
        self.assertNotEqual(proc.returncode, 0, f"{env} was accepted")
        self.assertIn(needle, proc.stderr, env)

    def accepted(self, **env: str) -> subprocess.CompletedProcess[str]:
        proc = _run_sh(**env)
        self.assertEqual(proc.returncode, 0, f"{env}: {proc.stderr}")
        return proc

    def test_defaults_pass(self) -> None:
        out = self.accepted().stdout
        d = _defaults()
        self.assertTrue(out.startswith("==> validate-only "), out)
        self.assertEqual(len(out.splitlines()), 1, out)
        for want in (f"tf={TF_SHA}", f"repo={TF_REPO}", f"snapshot={SNAPSHOT_SHA}",
                     f"tp=2 ctx={d['CONTEXT']} drafts={d['MTP_DRAFTS']}", f"confidence={d['MTP_CONFIDENCE']}", "thinking=0",
                     f"max_tokens={d['MAX_TOKENS']}", "host=0.0.0.0", f"mem_gate={d['MEM_GATE_GIB']}",
                     "engine_env=0", f"image={d['IMAGE']}"):
            self.assertIn(want, out)

    def test_image_tag_names_the_engine_commit(self) -> None:
        d = _defaults()
        self.assertTrue(d["IMAGE"].endswith("-" + d["TF_SHA"][:7]), d["IMAGE"])

    def test_validate_only_needs_no_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _stubs(Path(tmp), docker="echo docker called >&2; exit 99\n", ssh="echo ssh called >&2; exit 99\n",
                          python3="echo python called >&2; exit 99\n")["PATH"]
            for mode in ("1", "args"):
                env = _clean_env(PATH=path, VALIDATE_ONLY=mode)
                proc = subprocess.run(["bash", str(ROOT / "run.sh")], capture_output=True, text=True, env=env, check=False)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertNotIn("called", proc.stderr)
        bad = _run_sh(validate="2")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("must be 0, 1 or args", bad.stderr)

    def test_integers_are_decimal(self) -> None:
        self.refused("not a positive decimal integer", MAX_TOKENS="04096")
        self.refused("not a positive decimal integer", CONTEXT="8x")
        self.refused("not a positive decimal integer", READY_TIMEOUT="-1")
        self.refused("not a positive decimal integer", SHARDS="048")
        self.refused("not a non-negative decimal integer", MEM_GATE_GIB="0100")
        self.refused("not a non-negative decimal integer", MTP_DRAFTS="-1")
        self.refused("must be 0 or 1", BENCH_ONLY="yes")
        self.refused("must be 0 or 1", THINKING="on")
        self.refused("must be 0 or 1", IMAGE_ONLY="yes")
        self.refused("must be an integer in [-1000, 1000]", OOM_SCORE_ADJ="1001")
        self.accepted(CONTEXT="1048576", MTP_DRAFTS="0", OOM_SCORE_ADJ="-1000")

    def test_topology_window_and_drafts(self) -> None:
        self.refused("two ranks (one per Spark) only", TP="1")
        self.refused("exceeds the native window 1048576", CONTEXT="1048577")
        self.refused("exceeds 5, the most DSpark drafts a round", MTP_DRAFTS="6")
        self.refused("must be empty or a decimal in (0, 1]", MTP_CONFIDENCE="1.5")
        self.refused("must be empty or a decimal in (0, 1]", MTP_CONFIDENCE="0")
        self.refused("must be empty or a decimal in (0, 1]", MTP_CONFIDENCE=".7")
        self.assertIn("confidence=0.70", self.accepted(MTP_CONFIDENCE="0.70").stdout)
        self.refused("collides with MASTER_PORT", PORT=_defaults()["MASTER_PORT"])
        self.refused("at most 65535", MASTER_PORT="65536")

    def test_top_p(self) -> None:
        # TOP_P is the server's top_p when a request sends none: a decimal in (0, 1].
        self.assertEqual(_defaults()["TOP_P"], "1.0")
        self.assertIn("top_p=1.0", self.accepted().stdout)
        for bad in ("0", "1.5", ".9", "0.0", "1e-1", "0.95 "):
            self.refused("must be a decimal in (0, 1]", TOP_P=bad)
        self.assertIn("top_p=0.95", self.accepted(TOP_P="0.95").stdout)

    def test_memory_gate_floor(self) -> None:
        self.refused("is below 92 GiB", MEM_GATE_GIB="64")
        self.refused("is below 92 GiB", MEM_GATE_GIB="0")
        self.assertIn("mem_gate=0", self.accepted(MEM_GATE_GIB="0", FORCE_UNSAFE_MEM_GATE="1").stdout)
        self.assertIn("mem_gate=110", self.accepted(MEM_GATE_GIB="110").stdout)

    def test_modes(self) -> None:
        self.refused("ORCHESTRATE=yes must be auto or 0", ORCHESTRATE="yes")

    def test_container_name_keeps_the_recipe_prefix(self) -> None:
        # run.sh and stop.sh stop and remove this name: a foreign name must never get that far.
        for name in ("qwen38-flash-next-nvfp4", "tf-dsv41", "tf-dsv41-a.b", "TF-DSV41-x"):
            self.refused("must match tf-dsv41-[a-z0-9-]+", CONTAINER_NAME=name)
        self.accepted(CONTAINER_NAME="tf-dsv41-test-2")
        env = _clean_env(CONTAINER_NAME="qwen38-flash-next-nvfp4", ORCHESTRATE="0")
        proc = subprocess.run([str(ROOT / "stop.sh")], capture_output=True, text=True, env=env, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("must match tf-dsv41-[a-z0-9-]+", proc.stderr)

    def test_exported_engine_vars_must_be_recipe_knobs(self) -> None:
        self.refused("TF_DSV41_MODEL is exported but is not a recipe.yaml knob", TF_DSV41_MODEL="/x")
        out = _run_sh(validate="args", TF_DSV41_MODEL="/x", EXTRA_ENV="TF_DSV41_MODEL=/x").stdout
        self.assertIn("env: TF_DSV41_MODEL=/x", out)
        out = self.accepted(TF_DSV41_CACHE_GIB="4.5", TF_DSV41_CACHE_ENTRIES="4").stdout
        self.assertIn("engine_env=2", out)

    def test_engine_knob_values(self) -> None:
        self.refused("has a space or a quote", TF_DSV41_CACHE_GIB="3 4")
        self.refused("must be empty or a number of GiB", TF_DSV41_CACHE_GIB="-1")
        self.refused("must be empty or a non-negative integer", TF_DSV41_CACHE_ENTRIES="8.5")
        self.assertIn("env: TF_DSV41_CACHE_GIB=0\n", _run_sh(validate="args", TF_DSV41_CACHE_GIB="0").stdout)

    def test_paths_and_shas(self) -> None:
        self.refused("not a 40-hex TensorFold commit", TF_SHA="903a1e8")
        self.refused("must be an https git URL ending .git", TF_REPO="git@github.com:sfxnz/TensorFold.git")
        self.refused("not a 40-hex snapshot revision", SNAPSHOT_SHA="main")
        self.refused("not a 64-hex sha256", CONFIG_SHA256="6469adab")
        self.refused("not a 64-hex sha256", INDEX_SHA256="91731e4a")
        self.refused("must be an absolute path", TF_CACHE="cache/tf")
        self.refused("must be an absolute path", HF_CACHE="/a:/b")
        self.accepted(TF_REPO="https://github.com/ashhart/TensorFold.git")

    def test_served_name(self) -> None:
        self.refused("must be one word", SERVED_NAME="a'b")
        self.refused("must be one word", SERVED_NAME="a b")

    def test_extra_args_cannot_reset_owned_flags(self) -> None:
        for flag in ("--tp", "--context=8192", "--mtp-drafts", "--mtp-confidence", "--no-drafts", "--max-tokens",
                     "--port", "--name", "--master", "--master-port", "--host", "--thinking", "--no-thinking",
                     "--top-p", "--top-p=0.9"):
            self.refused("which run.sh passes itself", EXTRA_ARGS=f"{flag} 1")
        for flag in ("--parallel", "--prefill-fp8", "--drafter", "--thinking-budget", "--vision", "--kv-dtype"):
            self.refused("refuses or ignores it", EXTRA_ARGS=f"{flag} 1")
        self.accepted(EXTRA_ARGS="--temperature 0.6 --top-k 20")

    def test_extra_args_prefixes_are_refused(self) -> None:
        # TensorFold's argparse expands unambiguous prefixes: --cont is --context.
        for w in ("--cont 8", "--hos 0.0.0.0", "--mtp-d=1", "--vis", "--paral 2"):
            self.refused("EXTRA_ARGS sets", EXTRA_ARGS=w)

    def test_extra_env(self) -> None:
        self.refused("is not KEY=VALUE", EXTRA_ENV="NCCL_PROTO")
        self.refused("which run.sh sets itself", EXTRA_ENV="TF_DSV41_CACHE_GIB=4")
        self.refused("which run.sh sets itself", EXTRA_ENV="NCCL_IB_HCA=rocep1s0f1")
        self.refused("which run.sh sets itself", EXTRA_ENV="TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1")
        self.refused("which run.sh sets itself", EXTRA_ENV="CONTEXT=1")
        out = _run_sh(validate="args", EXTRA_ENV="NCCL_PROTO=LL TENSORFOLD_MEMORY_RESERVE_GIB=8").stdout
        self.assertIn("env: NCCL_PROTO=LL", out)
        self.assertIn("env: TENSORFOLD_MEMORY_RESERVE_GIB=8", out)

    def test_bench_only_binds_loopback(self) -> None:
        self.assertIn("host=127.0.0.1", self.accepted(BENCH_ONLY="1").stdout)
        self.assertIn("--host 127.0.0.1", _run_sh(validate="args", BENCH_ONLY="1").stdout)

    def test_every_refusal_precedes_validate_only_exit(self) -> None:
        run = _read("run.sh")
        exit_at = run.index('if [[ "${VALIDATE_ONLY:-0}" != 0 ]]')
        for needle in ('die "TP=', "exceeds the native window", "is below $MIN_MEM_GATE_GIB", "which run.sh passes itself",
                       "refuses or ignores it", "which run.sh sets itself", "has a space or a quote", "not a 64-hex"):
            self.assertLess(run.index(needle), exit_at, needle)


class ImagePinTests(unittest.TestCase):
    def test_tensorfold_pin_everywhere(self) -> None:
        run, dockerfile = _read("run.sh"), _read("docker/Dockerfile")
        self.assertIn(f"ARG TF_SHA={TF_SHA}\n", dockerfile)
        self.assertIn(f"ARG TF_REPO={TF_REPO}\n", dockerfile)
        self.assertIn(f'TF_SHA="${{TF_SHA:-{TF_SHA}}}"', run)
        self.assertIn(f'TF_REPO="${{TF_REPO:-{TF_REPO}}}"', run)
        # Every other sha256 written in the docs is a pin.
        pins = {BASE_DIGEST.split(":")[1], CONFIG_SHA256, INDEX_SHA256, BENCH_DECODE_SHA256}
        for rel in ("NOTICE", "README.md", "AGENTS.md", "docker/Dockerfile", "docker/constraints.txt"):
            for sha in re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", _read(rel)):
                self.assertIn(sha, pins, rel)
        if yaml is None:
            self.skipTest("PyYAML not installed")
        r = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)
        self.assertEqual(r["engine"]["commit"], TF_SHA)
        self.assertEqual(r["engine"]["repo"], TF_REPO)
        self.assertEqual(r["serve"]["env"]["TF_SHA"], TF_SHA)
        self.assertEqual(r["model"]["revision"], SNAPSHOT_SHA)
        self.assertEqual(r["serve"]["env"]["CONFIG_SHA256"], CONFIG_SHA256)
        self.assertEqual(r["serve"]["env"]["INDEX_SHA256"], INDEX_SHA256)

    def test_dockerfile_build_rules(self) -> None:
        d = _read("docker/Dockerfile")
        code = "\n".join(line for line in d.splitlines() if not line.startswith("#"))
        self.assertIn(f"ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3@{BASE_DIGEST}\n", d)
        if yaml is not None:
            image = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)["image"]
            self.assertEqual(f"{image['base']}@{image['digest']}", f"nvcr.io/nvidia/pytorch:26.07-py3@{BASE_DIGEST}")
        self.assertIn("FROM ${BASE}", d)
        # pip installs the pinned commit from git, under the constraints, and the commit it recorded is checked.
        self.assertIn('pip install --no-cache-dir -c /opt/tensorfold-constraints.txt "tensorfold @ git+${TF_REPO}@${TF_SHA}"', d)
        self.assertRegex(d, r"assert d\['vcs_info'\]\['commit_id'\] == '\$\{TF_SHA\}', d;")
        self.assertRegex(d, r"assert d\['url'\] == '\$\{TF_REPO\}', d;")
        # No step of the RUN chain may swallow its own failure.
        self.assertNotRegex(code, r"\|\|\s*(true|:)\b")
        # No installed package may change: torch and triton stay the base's.
        self.assertIn("diff /opt/base-freeze.txt -", d)
        self.assertIn("LABEL tensorfold.sha=${TF_SHA}", d)
        # /bin/sh is dash: `command -v a b c` checks only a. One name a call.
        self.assertIn('for b in tensorfold ninja nvcc; do command -v "$b" || exit 1; done', d)
        self.assertNotRegex(code, r"command -v [A-Za-z\"$]\S* [A-Za-z]")
        self.assertNotIn("HF_TOKEN=", code)
        cons = [l for l in _read("docker/constraints.txt").splitlines() if l and not l.startswith("#")]
        self.assertEqual(sorted(c.split("==")[0].lower() for c in cons),
                         ["huggingface-hub", "jinja2", "numpy", "safetensors", "tokenizers"])
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]+==[0-9.]+", c) for c in cons), cons)

    def test_run_sh_checks_the_image_label(self) -> None:
        run = _read("run.sh")
        self.assertIn('tensorfold.sha" }}', _func_body(run, "image_sha"))
        self.assertIn("{{.Id}}", _func_body(run, "sync_worker_image"))
        self.assertIn('docker save "$IMAGE" | ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" docker load',
                      _func_body(run, "sync_worker_image"))
        # ensure_image against a stub docker: LABEL is what `image inspect` prints, and a build sets it to BUILT.
        fns = (_func_src(run, "die") + "log() { echo \"==> $*\"; }\n" + _func_src(run, "image_sha")
               + _func_src(run, "ensure_image"))
        docker = ('echo "$*" >> "$STUB_LOG"\n'
                  'case "$1" in image) cat "$STUB_LABEL" 2>/dev/null;; build) echo "$BUILT" > "$STUB_LABEL";; esac\n')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "bin").mkdir()
            (tmp / "docker").mkdir()
            (tmp / "docker/Dockerfile").write_text("FROM x\n")
            env = _stubs(tmp / "bin", docker=docker)
            env.update(STUB_LOG=str(tmp / "calls"), STUB_LABEL=str(tmp / "label"))
            base = f"IMAGE=img TF_REPO={TF_REPO} TF_SHA={TF_SHA} SCRIPT_DIR={tmp}\n"
            # A label with another commit: refused, nothing built.
            (tmp / "label").write_text("1" * 40 + "\n")
            proc = _bash(fns + base + "ensure_image\n", env=env)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("carries TensorFold '" + "1" * 40 + "'", proc.stderr)
            self.assertNotIn("build", (tmp / "calls").read_text())
            # No image: built with both pins, then the new label is checked.
            (tmp / "label").unlink()
            env["BUILT"] = TF_SHA
            ok = _bash(fns + base + "ensure_image\n", env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn(f"build --build-arg TF_REPO={TF_REPO} --build-arg TF_SHA={TF_SHA} -t img {tmp}/docker",
                          (tmp / "calls").read_text())
            # A build whose label still differs is refused.
            (tmp / "label").unlink()
            env["BUILT"] = "2" * 40
            bad = _bash(fns + base + "ensure_image\n", env=env)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("carries TensorFold '" + "2" * 40 + "'", bad.stderr)


class RankIdentityTests(unittest.TestCase):
    """Both ranks get the same engine settings: one serve_args(), one container_env(), every setting forwarded."""

    def args_out(self, script: Path | None = None, **env: str) -> tuple[list[str], list[str], list[str]]:
        proc = _run_sh(validate="args", script=script, **env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertTrue(lines[0].startswith("rank1: ") and lines[1].startswith("rank0: "))
        r1 = shlex.split(lines[0].split(": ", 1)[1])
        r0 = shlex.split(lines[1].split(": ", 1)[1])
        envs = [l[len("env: "):] for l in lines[2:]]
        return r0, r1, envs

    @staticmethod
    def flags(argv: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        i = 3
        while i < len(argv):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.setdefault(argv[i], []).append(argv[i + 1])
                i += 2
            else:
                out.setdefault(argv[i], []).append("")
                i += 1
        return out

    def test_container_env(self) -> None:
        got = dict(e.split("=", 1) for e in self.args_out()[2])
        for k, v in (("NCCL_IB_HCA", "rocep1s0f1,roceP2p1s0f1"), ("NCCL_SOCKET_IFNAME", "enp1s0f1np1"),
                     ("HF_HUB_OFFLINE", "1"), ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0"),
                     ("TORCH_EXTENSIONS_DIR", "/cache/tf/torch_extensions"), ("TRITON_CACHE_DIR", "/cache/tf/triton")):
            self.assertEqual(got.get(k), v, k)
        self.assertFalse([k for k in got if k.startswith("TF_DSV41_")], "an empty knob must not be passed")
        envs = self.args_out(TF_DSV41_CACHE_GIB="2", TF_DSV41_CACHE_ENTRIES="4")[2]
        self.assertIn("TF_DSV41_CACHE_GIB=2", envs)
        self.assertIn("TF_DSV41_CACHE_ENTRIES=4", envs)
        self.assertEqual(len(envs), len(set(e.split("=", 1)[0] for e in envs)), "a key is passed twice")
        if yaml is None:
            self.skipTest("PyYAML not installed")
        serve_env = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)["serve"]["env"]
        self.assertEqual(sorted(k for k in serve_env if k.startswith("TF_DSV41_")),
                         ["TF_DSV41_CACHE_ENTRIES", "TF_DSV41_CACHE_GIB"])

    def test_rank_argv_differs_only_by_rank_and_http(self) -> None:
        r0, r1 = self.args_out()[:2]
        self.assertEqual(r0[:3], ["tensorfold", "serve", SNAP_IN_CONTAINER])
        self.assertEqual(r1[:3], ["tensorfold", "serve", SNAP_IN_CONTAINER])
        f0, f1 = self.flags(r0), self.flags(r1)
        self.assertEqual(f0.pop("--rank"), ["0"])
        self.assertEqual(f1.pop("--rank"), ["1"])
        self.assertEqual(set(f0) - set(f1), {"--host", "--port", "--max-tokens"})
        for k, v in f1.items():
            self.assertEqual(f0[k], v, k)
        self.assertEqual(f0["--name"], [SERVED])
        self.assertIn("--no-thinking", f1)
        self.assertEqual(f1["--mtp-drafts"], ["5"])
        self.assertEqual(f1["--mtp-confidence"], ["0.15"])
        self.assertEqual(f1["--top-p"], ["1.0"])
        self.assertNotIn("--temperature", f1)
        self.assertNotIn("--top-k", f1)
        for argv in self.args_out(TOP_P="0.95")[:2]:
            self.assertEqual(self.flags(argv)["--top-p"], ["0.95"])
        self.assertNotIn("--parallel", f1)
        self.assertNotIn("--kv-dtype", f1)
        thinking = self.flags(self.args_out(THINKING="1", MTP_CONFIDENCE="0.5")[1])
        self.assertIn("--thinking", thinking)
        self.assertNotIn("--no-thinking", thinking)
        self.assertEqual(thinking["--mtp-confidence"], ["0.5"])

    def test_receipt_flags_are_a_subset_of_the_default_argv(self) -> None:
        # The engine receipts' serve flags (environment.txt, both ranks) reach both ranks unchanged, except those
        # run.sh sets from its own variables.
        want = {k: v for k, v in self.flags(_receipt_serve_argv()).items() if k not in PER_BOOT_FLAGS}
        self.assertIn("--tp", want)
        self.assertIn("--no-thinking", want)
        for argv in self.args_out()[:2]:
            f = self.flags(argv)
            for k, v in want.items():
                self.assertEqual(f.get(k), v, k)
            self.assertEqual(f["--context"], [_defaults()["CONTEXT"]])

    def test_worker_copy_rebuilds_the_same_argv_and_env(self) -> None:
        # The head forwards worker_env over ssh to a bare /tmp copy of run.sh. Run that line in an empty env with
        # another HOME: the worker must print the same rank argv and the same container env as the head.
        overrides = {"TF_DSV41_CACHE_GIB": "2", "EXTRA_ENV": "NCCL_PROTO=LL A=$HOME", "MTP_DRAFTS": "2",
                     "HF_CACHE": "/srv/hf", "THINKING": "1"}
        head = self.args_out(**overrides)
        run = _read("run.sh")
        prefix = run[: run.index('if [[ "${VALIDATE_ONLY:-0}" != 0 ]]')]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "head").mkdir()
            (tmp / "head/run.sh").write_text(prefix + _func_src(run, "worker_env") + "worker_env\n")
            line = subprocess.run(["bash", str(tmp / "head/run.sh")], capture_output=True, text=True, check=True,
                                  env=_clean_env(**overrides)).stdout.strip()
            (tmp / "w").mkdir()
            shutil.copy(ROOT / "run.sh", tmp / "w/run.sh")
            proc = subprocess.run(["env", "-i", f"PATH={os.environ['PATH']}", "HOME=/nonexistent", "bash", "-c",
                                   f"{line} VALIDATE_ONLY=args bash {tmp}/w/run.sh"],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(line.startswith("ROLE=worker ORCHESTRATE=0 "), line)
        lines = proc.stdout.splitlines()
        self.assertEqual(shlex.split(lines[0].split(": ", 1)[1]), head[1])
        self.assertEqual(shlex.split(lines[1].split(": ", 1)[1]), head[0])
        self.assertEqual([l[len("env: "):] for l in lines[2:]], head[2])
        self.assertIn("A=$HOME", head[2])
        self.assertIn("/srv/hf", line)

    def test_forward_vars_cover_every_generated_setting(self) -> None:
        run = _read("run.sh")
        self.assertIn("FORWARD_VARS=(HF_HUB_DISABLE_XET FORCE_UNSAFE_MEM_GATE)", run)
        self.assertIn('[[ "$v" == ORCHESTRATE || "$v" == WORKER_HOST ]] || FORWARD_VARS+=("$v")', run)
        self.assertIn('[[ "$v" =~ ^TF_DSV41_ ]] && ENGINE_VARS+=("$v")', run)
        start = _func_body(run, "start_local")
        self.assertIn("done < <(container_env)", start)
        self.assertIn('mapfile -t args < <(serve_args "$rank")', start)


class SnapshotTests(unittest.TestCase):
    """check_snapshot: config and index sha256, shard count, total bytes and headers, on any node."""

    @staticmethod
    def _snap(root: Path, sizes: list[int]) -> tuple[Path, str, str, int]:
        snap = root / "snap"
        snap.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (snap / name).write_text("x")
        (snap / "config.json").write_text('{"model_type": "deepseek_v41"}')
        names = [f"model-{i:05d}-of-{len(sizes):05d}.safetensors" for i in range(1, len(sizes) + 1)]
        total = sum(_shard(snap / n, s) for n, s in zip(names, sizes))
        (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"t{i}": n for i, n in enumerate(names)}}))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        return snap, sha(snap / "config.json"), sha(snap / "model.safetensors.index.json"), total

    def check(self, snap: Path, cfg: str, idx: str, shards: int, total: int) -> subprocess.CompletedProcess[str]:
        fn = _func_src(_read("run.sh"), "check_snapshot")
        return _bash(fn + f"check_snapshot {shlex.quote(str(snap))} {cfg} {idx} {shards} {total}\n")

    def test_whole_snapshot_passes_and_each_defect_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap, cfg, idx, total = self._snap(Path(tmp), [64, 32, 16])
            ok = self.check(snap, cfg, idx, 3, total)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("3 shards", ok.stdout)
            self.assertIn("config.json (sha256", self.check(snap, "0" * 64, idx, 3, total).stderr)
            self.assertIn("model.safetensors.index.json (sha256", self.check(snap, cfg, "f" * 64, 3, total).stderr)
            self.assertIn("names 3 shards, pinned 48", self.check(snap, cfg, idx, 48, total).stderr)
            self.assertIn(f"shards hold {total} bytes, pinned {total + 1}", self.check(snap, cfg, idx, 3, total + 1).stderr)
            _shard(snap / "model-00002-of-00003.safetensors", 32, truncate=8)
            self.assertIn("model-00002-of-00003.safetensors (truncated)", self.check(snap, cfg, idx, 3, total).stderr)
            (snap / "model-00002-of-00003.safetensors").unlink()
            self.assertIn("model-00002-of-00003.safetensors (FileNotFoundError)", self.check(snap, cfg, idx, 3, total).stderr)
            _shard(snap / "model-00002-of-00003.safetensors", 32)
            (snap / "tokenizer.json").unlink()
            proc = self.check(snap, cfg, idx, 3, total)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("tokenizer.json", proc.stderr)

    def test_worker_check_runs_the_same_function_over_ssh(self) -> None:
        # check_worker_snapshot sends check_snapshot through `ssh … bash -s`; run that locally through a stub ssh.
        run = _read("run.sh")
        fns = (_func_src(run, "die") + _func_src(run, "check_snapshot") + _func_src(run, "rsync_hint")
               + _func_src(run, "check_worker_snapshot"))
        with tempfile.TemporaryDirectory() as tmp:
            snap, cfg, idx, total = self._snap(Path(tmp), [64, 32])
            (Path(tmp) / "bin").mkdir()
            env = _stubs(Path(tmp) / "bin", ssh='while [ "$1" = -o ]; do shift 2; done; shift; exec "$@"\n')
            base = (f"SNAPSHOT={shlex.quote(str(snap))} CONFIG_SHA256={cfg} INDEX_SHA256={idx} SHARDS=2 "
                    f"SHARD_BYTES={total} WORKER_HOST=w HEAD_IP=10.100.8.1\n")
            ok = _bash(fns + base + "check_worker_snapshot\n", env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("2 shards", ok.stdout)
            bad = _bash(fns + base + "SHARDS=48\ncheck_worker_snapshot\n", env=env)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("incomplete on w", bad.stderr)
            self.assertIn(f"rsync -aL --partial --mkpath 10.100.8.1:{snap}/ {snap}/", bad.stderr)

    def test_worker_never_downloads(self) -> None:
        run = _read("run.sh")
        fns = (_func_src(run, "die") + "log() { :; }\nhost_short() { echo spark2; }\nmkdir() { :; }\n"
               "snapshot_ok() { return 1; }\n" + _func_src(run, "hf_bin") + _func_src(run, "rsync_hint")
               + _func_src(run, "ensure_weights"))
        with tempfile.TemporaryDirectory() as tmp:
            env = _stubs(Path(tmp), hf='echo "hf $*" >&2; exit 0\n', df='echo Avail; echo "$STUB_FREE"\n',
                         du='printf "%s\\t%s\\n" "$STUB_HAVE" "$3"\n')
            env.update(STUB_FREE=str(400 << 30), STUB_HAVE="0")
            snap = "/c/hub/models--a--b/snapshots/abc"
            base = f"SKIP_DOWNLOAD=0 HEAD_IP=10.100.8.1 SNAPSHOT={snap} SHARD_BYTES={333 << 30}\n"
            proc = _bash(fns + base + "ROLE=worker\nensure_weights\n", env=env)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("the worker does not download", proc.stderr)
            # Only the pinned snapshot crosses the link, links resolved; never the whole cache entry.
            self.assertIn(f"(333 GiB free needed): rsync -aL --partial --mkpath 10.100.8.1:{snap}/ {snap}/", proc.stderr)
            self.assertNotIn("hf download", proc.stderr)
            head_env = "ROLE=head MODEL=a/b SNAPSHOT_SHA=abc HF_CACHE=/c HF_HUB_DISABLE_XET=1\nensure_weights\n"
            head = _bash(fns + base + head_env, env=env)
            self.assertIn("hf download a/b --revision abc --cache-dir /c/hub", head.stderr)
            # Too little free space: refused before the download; what blobs/ already holds counts.
            env["STUB_FREE"] = str(300 << 30)
            short = _bash(fns + base + head_env, env=env)
            self.assertNotEqual(short.returncode, 0)
            self.assertIn("/c has 300 GiB free; the download needs 334 GiB more", short.stderr)
            self.assertNotIn("hf download", short.stderr)
            env["STUB_HAVE"] = str(100 << 30)
            self.assertIn("hf download", _bash(fns + base + head_env, env=env).stderr)

    def test_worker_refusals_run_over_ssh_and_stop_nothing(self) -> None:
        # check_worker_free sends the refusals through `ssh … bash -s`; run that locally through stub ssh and docker.
        run = _read("run.sh")
        fns = (_func_src(run, "die") + _func_src(run, "host_short") + _func_src(run, "ours_names")
               + _func_src(run, "refuse_foreign_serve") + _func_src(run, "refuse_gpu_busy")
               + _func_src(run, "check_worker_free"))
        docker = ('echo "$*" >> "$STUB_LOG"\n'
                  'case "$*" in *label=*) printf "%s\\n" $OURS;; ps*) printf "%s\\n" $RUNNING;; '
                  'inspect*) echo \'[{"Driver":"nvidia"}]\';; esac\n')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = _stubs(tmp, ssh='while [ "$1" = -o ]; do shift 2; done; shift; exec "$@"\n', docker=docker,
                         hostname="echo spark2\n", **{"nvidia-smi": 'printf "%s\\n" $PIDS\n'})
            env.update(STUB_LOG=str(tmp / "calls"), OURS="", RUNNING="", PIDS="")
            base = "RECIPE_LABEL=ai-lab.recipe=dsv41-tensorfold CONTAINER_NAME=tf-dsv41-flash WORKER_HOST=spark2\n"
            ok = _bash(fns + base + "check_worker_free\n", env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            for running, ours, pids, needle in (("vllm-dsv41", "", "", "vllm-dsv41 is using GPUs"),
                                                ("", "", "4242", "A CUDA process is running on spark2")):
                env.update(RUNNING=running, OURS=ours, PIDS=pids)
                bad = _bash(fns + base + "check_worker_free\n", env=env)
                self.assertNotEqual(bad.returncode, 0)
                self.assertIn(needle, bad.stderr)
                self.assertIn("spark2 is not free for this recipe", bad.stderr)
            # Our own rank still holding the GPU there is left to its start_local.
            env.update(RUNNING="tf-dsv41-flash", OURS="tf-dsv41-flash", PIDS="4242")
            self.assertEqual(_bash(fns + base + "check_worker_free\n", env=env).returncode, 0)
            self.assertNotRegex((tmp / "calls").read_text(), r"(?m)^(rm|stop|kill)")

    def test_snapshot_pins_match_the_family_docs(self) -> None:
        # docs/recipes/deepseek-v4.1-flash.md: 48 shards; the model pins in recipe.yaml render into run.sh.
        run = _read("run.sh")
        self.assertIn('SHARDS="${SHARDS:-48}"', run)
        self.assertIn('SNAPSHOT="${HF_CACHE}/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/${SNAPSHOT_SHA}"', run)


class RunOpsTests(unittest.TestCase):
    """Plumbing that only runs on a real boot, checked statically or in isolation."""

    def test_memavail_and_mem_gate(self) -> None:
        run = _read("run.sh")
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemTotal: 127607404 kB\nMemFree:   74000000 kB\nMemAvailable: 120586240 kB\n")
            out = _bash(_func_src(run, "memavail_gib") + f"memavail_gib {shlex.quote(str(meminfo))}\n").stdout
            self.assertEqual(out, "115")
        fns = (_func_src(run, "die") + "log() { printf '==> %s\\n' \"$*\"; }\nhost_short() { echo n; }\n"
               + _func_src(run, "mem_gate") + "sleep() { :; }\n")
        ok = _bash(fns + "memavail_gib() { echo 115; }\nMEM_GATE_GIB=100 MEM_GATE_TIMEOUT=600\nmem_gate\n")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("MemAvailable 115 GiB >= 100", ok.stdout)
        short = _bash(fns + "memavail_gib() { echo 20; }\nMEM_GATE_GIB=100 MEM_GATE_TIMEOUT=1\nmem_gate\n")
        self.assertNotEqual(short.returncode, 0)
        self.assertIn("MemAvailable 20 GiB < MEM_GATE_GIB=100", short.stderr)
        off = _bash(fns + "memavail_gib() { echo 1; }\nMEM_GATE_GIB=0 MEM_GATE_TIMEOUT=1\nmem_gate\n")
        self.assertEqual(off.returncode, 0)
        self.assertNotIn("sudo", run)
        self.assertNotIn("drop_caches", run)

    def test_docker_run_flags_and_mounts(self) -> None:
        start = _func_body(_read("run.sh"), "start_local")
        for flag in ("--init", "--ulimit core=1", '--oom-score-adj "$OOM_SCORE_ADJ"', "--cap-add IPC_LOCK",
                     "--ulimit memlock=-1:-1", "--ulimit stack=67108864", "--device /dev/infiniband", "--network host",
                     "--ipc host", "--restart no", '-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro"',
                     '-v "${TF_CACHE}/${TF_SHA}:/cache/tf"',
                     'tensorfold serve "$SNAPSHOT_IN_CONTAINER" "${args[@]}" $EXTRA_ARGS >/dev/null\n  start_memguard'):
            self.assertIn(flag, start, flag)
        self.assertLess(start.index("\n  refuse_foreign_serve"), start.index("else prepare_node"))
        self.assertLess(start.index("else prepare_node"), start.index("docker run -d"))
        self.assertIn('--label "$RECIPE_LABEL"', start)
        self.assertNotIn("HF_TOKEN", _read("run.sh"))

    def test_prepare_node_order(self) -> None:
        body = _func_body(_read("run.sh"), "prepare_node")
        steps = ["mkdir -p", "ensure_image", "ensure_weights", "check_hca", "refuse_gpu_busy", "mem_gate"]
        at = [body.index(f"\n  {s}") for s in steps]
        self.assertEqual(at, sorted(at))

    def test_head_orchestration_order(self) -> None:
        run = _read("run.sh")
        block = run[run.index('if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]'):]
        scp = block.index("scp ")
        steps = ["take_lock", "refuse_foreign_serve", "refuse_busy_ports", "require_worker_ssh", "check_worker_free",
                 "ensure_weights", "check_worker_snapshot", "ensure_image", "sync_worker_image", "prepare_node"]
        at = [block.index(f"\n  {s}\n") for s in steps]
        # The worker's refusals before anything long; the head's weights before the worker's copy is checked (the
        # worker rsyncs from the head); the image before any memory gate; all of it before rank 1 starts.
        self.assertEqual(at, sorted(at), steps)
        self.assertLess(at[-1], scp)
        # Rank 1 first: the worker's start and its state check come before rank 0's start.
        worker_start = block.index('ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "$(worker_env) bash')
        self.assertLess(scp, worker_start)
        self.assertLess(worker_start, block.index('state="$(worker_state)"'))
        self.assertLess(block.index('state="$(worker_state)"'), block.index("start_local 0 prepared"))
        self.assertIn("scp -q -o BatchMode=yes -o ConnectTimeout=5", block)
        self.assertIn("Refusing to start a TP=2 head rank alone", _func_body(run, "require_worker_ssh"))
        # From before rank 1 starts until rank 0 answers, any exit stops both ranks; prepare_node runs once.
        trap = block.index("trap 'fail_stop")
        self.assertLess(block.index("prepare_node"), trap)
        self.assertLess(trap, scp)
        self.assertIn("start_local 0 prepared", block)
        self.assertLess(block.index("start_local 0 prepared"), block.index("wait_ready 1"))
        self.assertLess(block.index("wait_ready 1"), block.index("trap - EXIT"))
        self.assertTrue(_func_body(run, "fail_stop").lstrip().startswith("trap - EXIT"))
        self.assertIn('"$SCRIPT_DIR/stop.sh"', _func_body(run, "fail_stop"))
        start = _func_body(run, "start_local")
        self.assertIn('if [[ "$prepared" == prepared ]]; then mem_gate; else prepare_node; fi', start)
        self.assertIn("flock -n 9", _func_body(run, "take_lock"))
        self.assertIn("9>&-", _func_body(run, "start_memguard"))
        wait = _func_body(run, "wait_ready")
        for needle in ("$SERVED_NAME", "/health", "worker_state", "abort_worker_dead", "READY_TIMEOUT", "startup estimate"):
            self.assertIn(needle, wait)

    def test_image_only_builds_and_ships_without_gpu_or_gate(self) -> None:
        run = _read("run.sh")
        block = run[run.index('if [[ "$IMAGE_ONLY" == 1 ]]; then'):]
        block = "\n".join(l for l in block[: block.index("\nfi\n")].splitlines() if not l.lstrip().startswith("#"))
        self.assertLess(run.index('if [[ "$IMAGE_ONLY" == 1 ]]; then'),
                        run.index('if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]'))
        for needle in ("take_lock", "ensure_image", "sync_worker_image", "exit 0"):
            self.assertIn(needle, block)
        self.assertLess(block.index("take_lock"), block.index("ensure_image"))
        for needle in ("prepare_node", "mem_gate", "refuse_foreign_serve", "start_local", "check_snapshot"):
            self.assertNotIn(needle, block)

    def test_worker_state_reads_a_missing_container(self) -> None:
        fn = _func_src(_read("run.sh"), "worker_state")
        for ssh_out, rc, want in (("\\nmissing\\n", 0, "missing"), ("false\\n", 0, "false"), ("", 255, "")):
            with tempfile.TemporaryDirectory() as tmp:
                env = _stubs(Path(tmp), ssh=f"printf '{ssh_out}'\nexit {rc}\n")
                out = _bash(f"set -euo pipefail\n{fn}WORKER_HOST=w CONTAINER_NAME=c\nprintf '[%s]' \"$(worker_state)\"\n", env=env)
                self.assertEqual(out.stdout, f"[{want}]")

    def test_memguard_loop_kills_when_ram_and_swap_are_low(self) -> None:
        body = _func_src(_read("run.sh"), "memguard_loop")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls"
            env = _stubs(Path(tmp), docker='echo "$@" >> "$STUB_LOG"\necho true\n', logger="exit 0\n", sleep="exit 0\n")
            env["STUB_LOG"] = str(log)
            proc = _bash(body + "\nmemguard_loop tf-test 99999999 99999999", env=env, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("kill tf-test", log.read_text())

    def test_no_grep_q_at_the_end_of_a_pipe(self) -> None:
        # Under pipefail a `grep -q` that exits on its first match can SIGPIPE the writer and fail the pipe: a found
        # container reads as gone, a foreign GPU as absent. Here-strings only.
        for rel in ("run.sh", "stop.sh", "tools/session_gate.sh"):
            self.assertNotRegex(_read(rel), r"\|\s*grep\s+-[A-Za-z]*q", rel)

    def test_port_check_needs_ss(self) -> None:
        self.assertIn("command -v ss", _func_body(_read("run.sh"), "refuse_busy_ports"))

    def test_hca_check(self) -> None:
        run = _read("run.sh")
        self.assertIn("/sys/class/infiniband/$h/ports/1/state", _func_body(run, "check_hca"))
        self.assertIn("ACTIVE", _func_body(run, "check_hca"))

    def test_worker_state_dir_is_under_its_cache(self) -> None:
        run = _read("run.sh")
        self.assertIn('if [[ "${ROLE:-}" == worker ]]; then STATE_DIR="$TF_CACHE/run-state"; fi', run)
        self.assertLess(run.index('STATE_DIR="$TF_CACHE/run-state"'), run.index('ROLE="$(detect_role)"'))

    def test_run_state_under_script_dir(self) -> None:
        for rel in ("run.sh", "stop.sh"):
            text = _read(rel)
            self.assertNotIn("${PWD}/.run-state", text, rel)
            self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', text, rel)
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_stop_is_graceful_head_first_and_ours_only(self) -> None:
        stop = _read("stop.sh")
        self.assertLess(stop.index('\nstop_ours "$CONTAINER_NAME" "$STOP_TIMEOUT" 0\n'), stop.index('ssh "$WORKER_HOST"'))
        self.assertIn('stop_ours $(printf %q "$CONTAINER_NAME") $STOP_TIMEOUT 1', stop)
        self.assertRegex(stop, r"ssh -o BatchMode=yes.*\n(?:.*\n)*?\s+else\n(?:.*\n)*?\s+exit 1")
        self.assertEqual(set(re.findall(r"docker (?:stop|rm -f|kill)[^\n;']*", stop)),
                         {'docker stop -t "$t" "$name" >/dev/null 2>&1 || true', 'docker rm -f "$name" >/dev/null'})
        label = re.search(r"^RECIPE_LABEL=(\S+)$", _read("run.sh"), re.M).group(1)
        self.assertIn(f"--filter label={label} ", stop)

    def test_stop_removes_only_labelled_containers(self) -> None:
        # docker ps --filter label=… lists only this recipe's containers; a same-named foreign one is left alone.
        for script, call in (("stop.sh", None), ("run.sh", "stop_local")):
            with tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "calls"
                docker = ('echo "$*" >> "$STUB_LOG"\n'
                          'case "$*" in *"label=ai-lab.recipe=dsv41-tensorfold"*) printf "%s\\n" $OURS;; esac\n')
                env = _stubs(Path(tmp), docker=docker, sleep="exit 0\n")
                env.update(STUB_LOG=str(log), ORCHESTRATE="0", CONTAINER_NAME="tf-dsv41-flash")
                for ours, removed in (("", []), ("tf-dsv41-flash other", ["tf-dsv41-flash"])):
                    log.write_text("")
                    env["OURS"] = ours
                    if call is None:
                        proc = subprocess.run([str(ROOT / "stop.sh")], capture_output=True, text=True, env=env, check=False)
                    else:
                        run = _read("run.sh")
                        fns = ("log() { :; }\nRECIPE_LABEL=" + re.search(r"^RECIPE_LABEL=(\S+)$", run, re.M).group(1) + "\n"
                               + _func_src(run, "ours_names") + _func_src(run, "stop_local") + "stop_local\n")
                        proc = _bash(fns, env=env)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    got = re.findall(r"^rm -f (\S+)$", log.read_text(), re.M)
                    self.assertEqual(got, removed, (script, ours))

    def test_foreign_gpu_container_is_refused_never_removed(self) -> None:
        run = _read("run.sh")
        fns = _func_src(run, "die") + "host_short() { echo n; }\n" + _func_src(run, "ours_names") + _func_src(run, "refuse_foreign_serve")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls"
            docker = ('echo "$*" >> "$STUB_LOG"\n'
                      'case "$*" in ps\\ -a*) ;; ps*) echo vllm-dsv41;; inspect*) echo \'[{"Driver":"nvidia"}]\';; esac\n')
            env = _stubs(Path(tmp), docker=docker)
            env["STUB_LOG"] = str(log)
            proc = _bash(fns + "RECIPE_LABEL=ai-lab.recipe=dsv41-tensorfold CONTAINER_NAME=tf-dsv41-flash\nrefuse_foreign_serve\n", env=env)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("vllm-dsv41 is using GPUs", proc.stderr)
            self.assertNotRegex(log.read_text(), r"^(rm|stop|kill)", log.read_text())

    def test_stop_sh_fails_loudly_without_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _stubs(Path(tmp), docker="exit 0\n", ssh="exit 255\n", hostname="echo spark1\n")
            env["WORKER_HOST"] = "nowhere"
            proc = subprocess.run([str(ROOT / "stop.sh")], capture_output=True, text=True, env=env, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Cannot SSH to nowhere", proc.stderr)


@unittest.skipIf(yaml is None, "PyYAML not installed")
class RenderTests(unittest.TestCase):
    def test_generated_blocks_are_current(self) -> None:
        proc = subprocess.run([sys.executable, str(ROOT / "kit/render.py"), "--check"], capture_output=True, text=True,
                              check=False)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_measured_rows_follow_their_evidence(self) -> None:
        # Per row: no evidence -> "not yet measured" and no number; evidence -> the file exists and the row shows
        # its numbers, which the file's SUMMARY contains.
        recipe = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)
        readme = _read("README.md")
        block = readme.split("<!-- BEGIN generated measured", 1)[1].split("<!-- END generated measured -->", 1)[0]
        rows = [l for l in block.splitlines() if l.startswith("| ") and not l.startswith("| Phase")]
        want = recipe["measured"]["decode"]["rows"]
        self.assertEqual(len(rows), len(want))
        for row, r in zip(rows, want):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if r["evidence"] in ("", "null", "~"):
                self.assertEqual(cells[2:], ["not yet measured on this pair", "–", "–"], row)
                continue
            self.assertTrue((ROOT / r["evidence"]).is_file(), r["evidence"])
            self.assertEqual(cells[2:], [r["decode"], r["aggregate"], f"{r['ttft_p50']} s"], row)
            summary = json.loads(_read(r["evidence"]).split("SUMMARY ", 1)[1])
            cell = next(c for c in summary if c["phase"] == r["phase"] and str(c["concurrency"]) == r["concurrency"])
            self.assertEqual(f"{cell['median_decode_tok_s']:.1f}", r["decode"])
            self.assertEqual(f"{cell['median_agg_tok_s']:.1f}", r["aggregate"])
            self.assertEqual(f"{cell['ttft_s_p50']:.2f}", r["ttft_p50"])

    @staticmethod
    def _render(edit=None, block_line: str | None = None, check: bool = True,
                evidence: str | None = None) -> subprocess.CompletedProcess[str]:
        """kit/render.py on a scratch copy; `edit` changes the parsed recipe, `block_line` goes into the run.sh block."""
        recipe = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)
        if edit:
            edit(recipe)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "kit").mkdir()
            shutil.copy(ROOT / "kit/render.py", tmp / "kit/render.py")
            shutil.copy(ROOT / "README.md", tmp / "README.md")
            shutil.copytree(ROOT / "evidence", tmp / "evidence")
            run = _read("run.sh")
            if block_line is not None:
                run = run.replace("# END generated\n", f"{block_line}\n# END generated\n", 1)
            (tmp / "run.sh").write_text(run)
            (tmp / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
            if evidence:
                (tmp / evidence).parent.mkdir(parents=True, exist_ok=True)
                (tmp / evidence).write_text("{}")
            proc = subprocess.run([sys.executable, str(tmp / "kit/render.py")] + (["--check"] if check else []),
                                  capture_output=True, text=True, check=False)
            proc.readme = (tmp / "README.md").read_text()  # type: ignore[attr-defined]
        return proc

    @staticmethod
    def _row(**kv: str):
        def edit(recipe) -> None:
            recipe["measured"]["decode"]["rows"][0].update(kv)
        return edit

    def test_render_fixture_round_trips(self) -> None:
        self.assertEqual(self._render().returncode, 0)

    def test_a_number_without_evidence_is_refused(self) -> None:
        proc = self._render(self._row(decode="44.2", evidence="null"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("has a number but no evidence file", proc.stderr)

    def test_a_row_without_evidence_renders_not_yet_measured(self) -> None:
        proc = self._render(self._row(decode="null", aggregate="null", ttft_p50="null", evidence="null", note=""),
                            check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("| prose | 1 | not yet measured on this pair | – | – |", proc.readme)

    def test_a_missing_evidence_file_is_refused(self) -> None:
        proc = self._render(self._row(decode="99.9", aggregate="99.9", ttft_p50="0.1",
                                      evidence="evidence/s9-made-up/nothing.json"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("does not exist", proc.stderr)

    def test_evidence_needs_all_three_numbers(self) -> None:
        ev = "evidence/s9/bench.json"
        proc = self._render(self._row(decode="null", evidence=ev), evidence=ev)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not all of decode, aggregate and ttft_p50", proc.stderr)

    def test_hand_lines_in_the_run_sh_block_are_refused(self) -> None:
        for line in ("MEMGUARD=0", "MEM_GATE_TIMEOUT=1", "export HF_CACHE=/tmp/hf", "CONTEXT=1048576"):
            proc = self._render(block_line=line)
            self.assertNotEqual(proc.returncode, 0, line)
            self.assertIn("inside the generated block is not", proc.stderr)
        self.assertEqual(self._render(block_line="# a comment").returncode, 0)

    def test_generated_names_are_assigned_only_on_their_line(self) -> None:
        # Outside the generated block nothing re-assigns a recipe value.
        run = _read("run.sh")
        lines = run.splitlines()
        b, e = lines.index(next(l for l in lines if l.startswith("# BEGIN generated"))), lines.index("# END generated")
        for name in _generated_names():
            for i, line in enumerate(lines):
                if b < i < e or not re.match(rf"\s*(export\s+)?{name}=", line):
                    continue
                same = (f'{name}="${name}"', f'{name}=$(printf %q "${name}")')  # passes the value on unchanged
                self.assertTrue(line.lstrip().startswith(same), f"run.sh:{i + 1} sets {name}: {line}")

    def test_context_names_its_status(self) -> None:
        # The comment above CONTEXT says "pending measurement" or names the evidence file of the measured window.
        m = re.search(r"((?:[ \t]+#[^\n]*\n)+)[ \t]+CONTEXT: (\d+)\n", _read("recipe.yaml"))
        self.assertIsNotNone(m)
        comment = m.group(1)
        if "pending measurement" not in comment:
            files = re.findall(r"evidence/\S+", comment)
            self.assertTrue(files, comment)
            for f in files:
                self.assertTrue((ROOT / f.rstrip(".,;)")).exists(), f)

    def test_measured_conditions_match_the_bench_log(self) -> None:
        decode = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)["measured"]["decode"]
        for ev in {r["evidence"] for r in decode["rows"] if r["evidence"] not in ("", "null", "~")}:
            header = _read(ev).splitlines()[0]
            runs = re.search(r"\bruns=(\d+)", header).group(1)
            max_tokens = re.search(r"\bmax_tokens=(\d+)", header).group(1)
            self.assertIn(f"max_tokens {max_tokens}, {runs} runs", decode["conditions"], ev)

    def test_prose_values_match_recipe_yaml(self) -> None:
        # Values written into README.md / AGENTS.md prose as NAME=value or `NAME` (value) equal recipe.yaml's.
        # Switches documented with their other value (ORCHESTRATE=0, BENCH_ONLY=1, ...) are left out.
        env = dict(yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)["serve"]["env"])
        env["STOP_TIMEOUT"] = re.search(r'STOP_TIMEOUT="\$\{STOP_TIMEOUT:-(\d+)\}"', _read("stop.sh")).group(1)
        switches = {"ORCHESTRATE", "BENCH_ONLY", "SKIP_DOWNLOAD", "THINKING", "MEMGUARD", "MTP_CONFIDENCE",
                    "EXTRA_ARGS", "EXTRA_ENV"}
        floor = re.search(r"^MIN_MEM_GATE_GIB=(\d+)$", _read("run.sh"), re.M).group(1)
        checked = 0
        for rel in ("README.md", "AGENTS.md"):
            text = _read(rel)
            for name, want in env.items():
                if name in switches or name.startswith("TF_DSV41_"):
                    continue
                pat = rf"(?<![A-Za-z0-9_]){name}=([^\s`;)]+)|`{name}` \(`?([^\s`)]+)"
                for m in re.finditer(pat, text):
                    got = (m.group(1) or m.group(2)).rstrip(".,")
                    if any(c in got for c in "…<${\"'"):
                        continue
                    self.assertEqual(got, want, f"{rel}: {m.group(0)}")
                    checked += 1
            for m in re.finditer(r"below (\d+)(?: GiB)?", text):
                self.assertEqual(m.group(1), floor, f"{rel}: {m.group(0)}")
        self.assertGreater(checked, 10)

    def test_defaults_in_other_scripts_match_recipe_yaml(self) -> None:
        env = yaml.load(_read("recipe.yaml"), Loader=yaml.BaseLoader)["serve"]["env"]
        self.assertIn(f'CONTAINER_NAME="${{CONTAINER_NAME:-{env["CONTAINER_NAME"]}}}"', _read("stop.sh"))
        gate = _read("tools/session_gate.sh")
        self.assertIn(f'CONTAINER_NAME="${{CONTAINER_NAME:-{env["CONTAINER_NAME"]}}}"', gate)
        self.assertIn(f'MODEL="${{MODEL:-{env["SERVED_NAME"]}}}"', gate)
        for rel in ("smoke_count.py", "smoke_chat.py", "bench_decode.py"):
            self.assertIn(f'default="{env["SERVED_NAME"]}"', _read(rel), rel)


class ToolsTests(unittest.TestCase):
    def test_verbatim_files_are_byte_identical(self) -> None:
        for rel, sha in VERBATIM.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), sha, rel)

    def test_ruler_sha_in_docs(self) -> None:
        self.assertIn(BENCH_DECODE_SHA256[:8], _read("README.md"))
        self.assertIn(BENCH_DECODE_SHA256[:8], _read("AGENTS.md"))

    def test_render_names_its_source(self) -> None:
        head = _read("kit/render.py").split("\nimport ", 1)[0]
        self.assertIn("kit/render.py @ 4eef42d", head)
        self.assertIn("sha256 f2aa3151e21687179142262545db93e142456cf83e74821fa32e84429a6466e1", head)

    def test_no_third_party_engine_left(self) -> None:
        # The recipe serves TensorFold's own deepseek_v41 family: no patches, no vendored launcher code, no repack.
        for rel in ("docker/patches", "docker/prebuild_ext.py", "tools/vendor", "tools/pack_engram.py", "LICENSES",
                    "evidence/s0-engram", "evidence/s0-upstream-image"):
            self.assertFalse((ROOT / rel).exists(), rel)
        needle = re.compile(r"jayleaton|miaai|mia\s?2\.9|2\.9 ?bpw|pack_engram|canary\.py|prebuild_ext", re.I)
        this = Path(__file__).resolve()
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT).as_posix()
            if not path.is_file() or path == this or rel.startswith((".git/", "evidence/")) or "__pycache__" in rel:
                continue
            self.assertIsNone(needle.search(path.read_text(errors="replace")), rel)

    def test_session_gate_pass_fail_and_receipts(self) -> None:
        # Stub the live serve (curl, docker, ssh, free) and the clients (python3); the gate's own plumbing runs.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env_lines = "HF_TOKEN=hf_secret\\nNCCL_IB_HCA=rocep1s0f1\\nTORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0\\n"
            stubs = {
                "curl": ('case "$*" in *http_code*) echo 200;; *v1/models*) '
                         f'echo \'{{"data":[{{"id":"{SERVED}"}}]}}\';; *) echo {{}};; esac\n'),
                "docker": (f'case "$*" in *Config.Env*) printf "{env_lines}";; *logs*) '
                           'echo "[tensorfold] CUDA rank 0 startup estimate 78.46 GiB within 99.76 GiB";; '
                           '*) echo img args;; esac\n'),
                "ssh": f'case "$*" in *Config.Env*) printf "{env_lines}";; *) echo remote;; esac\n',
                "free": "echo Mem: 1 2 3\n",
                "python3": ('echo "$*" >> "$STUB_LOG"\n'
                            'case "$1" in smoke_count.py) exit "${FAIL_COUNT:-0}";; esac\nexit 0\n'),
            }
            (tmp / "bin").mkdir()
            env = _stubs(tmp / "bin", **stubs)
            env["STUB_LOG"] = str(tmp / "py-calls")
            gate = ROOT / "tools/session_gate.sh"
            ok = subprocess.run([str(gate), str(tmp / "ev1")], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertEqual((tmp / "ev1/gate.txt").read_text().splitlines()[-1], "GATE=PASS")
            for name in ("smoke-chat", "smoke-count", "bench-frozen", "bench-prose-long"):
                self.assertEqual((tmp / f"ev1/{name}.exit").read_text().strip(), "0", name)
            calls = (tmp / "py-calls").read_text()
            self.assertIn("--phase both --concurrency 1 --max-tokens 200 --runs 9", calls)
            self.assertIn("--phase prose_long --concurrency 1 --max-tokens 200 --runs 5", calls)
            self.assertIn("startup estimate", (tmp / "ev1/startup.txt").read_text())
            hashed = (tmp / "ev1/harness.sha256").read_text()
            for rel in ("bench_decode.py", "run.sh", "stop.sh", "recipe.yaml"):
                self.assertIn(rel, hashed)
            for rank in (0, 1):
                text = (tmp / f"ev1/env-rank{rank}.txt").read_text()
                self.assertIn("HF_TOKEN=<redacted>", text)
                self.assertNotIn("hf_secret", text)
                self.assertIn("NCCL_IB_HCA=rocep1s0f1", text)
            env["FAIL_COUNT"] = "1"
            bad = subprocess.run([str(gate), str(tmp / "ev2")], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
            self.assertEqual((tmp / "ev2/gate.txt").read_text().splitlines()[-1], "GATE=FAIL")
            self.assertIn("smoke-count=1", (tmp / "ev2/gate.txt").read_text())

    def test_shell_tools_parse(self) -> None:
        for rel in ("run.sh", "stop.sh", "tools/session_gate.sh"):
            subprocess.run(["bash", "-n", str(ROOT / rel)], check=True)


if __name__ == "__main__":
    unittest.main()
