"""Nsight Compute step of the CUDA-Hercules verifier.

After a candidate compiles, passes correctness and the hacking audit, the verifier profiles the
candidate's own kernels with ncu and asks the workflow model for ranked next-step optimization
suggestions. Only those suggestions go back into the workflow (attempt memory -> coder /
replanner); the ncu reports and the analysis stay in the attempt's profile directory.

Collection and analysis follow KDA's ncu-report-skill (github.com/NVlabs/kda,
skills/ncu-report-skill): Recipe 1 (`--set full` + PM sampling), Recipe 2 (`--set source` +
SourceCounters, with -lineinfo), Recipe 3 (details page with the rule engine's estimated
speedups), per-line stall hotspots via the skill's extract_stall_hotspots.py helper, and the
skill's golden rule / analysis dimensions / diagnosis playbook as the analysis prompt.

Class 1 only for now: the task harness is rebuilt with -lineinfo in a temp copy and run with
KH_WARMUP=0 KH_ITERS=1; only kernels defined in the candidate's source are profiled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .llm import OpenAIChatClient

DETAILS_CHARS = 30000
HOTSPOT_CHARS = 8000
SUGGESTION_CHARS = 5000
MAX_KERNELS = 4

_KERNEL_RE = re.compile(
    r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:static\s+|inline\s+|__forceinline__\s+)*"
    r"(?:void|[\w:<>]+)\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)\s*\(")

# Runs in a fresh process: build the profiling binary in a temp copy of the task and collect.
_CHILD = r"""
import json, os, re, shlex, subprocess, sys
sys.path.insert(0, os.path.join({root!r}, "src"))
from cuda_hercules.runner import discover_tasks
from cuda_hercules.backend_base import resolve_backend
from cuda_hercules import backend_class1 as b1
out = {out!r}; ncu = {ncu!r}; regex = {regex!r}; count = {count!r}
res = {{"ok": False}}
cfg = {{t.task_id: t for t in discover_tasks({root!r})}}[{task_id!r}]
be = resolve_backend(cfg.runner.backend)
try:
    be.prepare(cfg, {sol!r})   # temp copy of the task with the candidate as the solution file
    run_dir = be._run_dir
    make_vars = b1._current_arch_make_vars(b1._get_gpu_sm()) if b1._is_class1_general_task(cfg) else {{}}
    mk = open(os.path.join(run_dir, "Makefile")).read()
    m = re.search(r"^NVCC_FLAGS\s*:?=\s*(.*)$", mk, re.M)
    flags = (m.group(1).strip() if m else "-O3") + " -lineinfo"
    target = shlex.split(cfg.build.cmd)[-1] if cfg.build.cmd.strip().startswith("make") else "test"
    make = ["make", "-B", target] + [f"{{k}}={{v}}" for k, v in make_vars.items()] + [f"NVCC_FLAGS={{flags}}"]
    rs = os.path.join({root!r}, "reference_sources")
    env = dict(os.environ, CUTLASS_DIR=os.path.join(rs, "cutlass"), TK_DIR=os.path.join(rs, "ThunderKittens"),
               DEEPGEMM_DIR=os.path.join(rs, "deep_gemm"), CUDNN_FE_DIR=os.path.join(rs, "cudnn_frontend"))
    p = subprocess.run(make, cwd=run_dir, capture_output=True, text=True, env=env, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError("profiling build failed: " + p.stderr[-1500:])
    exe = shlex.split(cfg.execute.cmd) if cfg.execute and cfg.execute.cmd else ["./test"]
    renv = dict(os.environ, KH_WARMUP="0", KH_ITERS="1")
    logs = {{}}
    for tag, sel in (("full", ["--set", "full", "--section", "PmSampling", "--section", "PmSampling_WarpStates"]),
                     ("source", ["--set", "source", "--section", "SourceCounters"])):
        cmd = [ncu, *sel, "-k", "regex:" + regex, "-c", str(count), "-f", "-o", os.path.join(out, tag), *exe]
        p = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, env=renv, timeout={timeout!r})
        logs[tag] = (p.stdout[-3000:] + p.stderr[-3000:])
    open(os.path.join(out, "ncu_log.txt"), "w").write(json.dumps(logs, indent=1))
    rep = os.path.join(out, "full.ncu-rep")
    if not os.path.isfile(rep):
        raise RuntimeError("ncu produced no report (no matching kernel launch?): " + logs.get("full", "")[-800:])
    p = subprocess.run([ncu, "--import", rep, "--page", "details"], capture_output=True, text=True, timeout=600)
    open(os.path.join(out, "details.txt"), "w").write(p.stdout)
    res = {{"ok": True}}
except Exception as e:
    res = {{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}
finally:
    try:
        be.cleanup(cfg)
    except Exception:
        pass
sys.stdout.write("__NCU_JSON__" + json.dumps(res) + "\n")
"""


def kernel_names(code: str) -> list[str]:
    names = []
    for m in _KERNEL_RE.finditer(code):
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


def _ncu_python_dir(ncu: str) -> str:
    real = os.path.realpath(shutil.which(ncu) or ncu)
    root = Path(real).resolve().parents[1]
    for cand in sorted(root.glob("nsight-compute*/extras/python")) + [Path(real).parent / "extras" / "python"]:
        if (cand / "ncu_report.py").is_file():
            return str(cand)
    return ""


class NcuProfiler:
    def __init__(self, *, root: str, skill_dir: str, llm: OpenAIChatClient, device: int | None,
                 timeout_s: int = 1200, ncu: str = "ncu"):
        self.root = root
        self.skill_dir = Path(skill_dir)
        if not (self.skill_dir / "SKILL.md").is_file():
            raise FileNotFoundError(f"ncu-report-skill not found under {skill_dir}")
        self.llm = llm
        self.device = device
        self.timeout_s = timeout_s
        self.ncu = ncu
        self.skill_prompt = "\n\n".join(
            (self.skill_dir / f).read_text(encoding="utf-8")
            for f in ("SKILL.md", "reference/05-analysis-dimensions.md", "reference/06-diagnosis-playbook.md"))

    def profile(self, problem, code_path: Path, out_dir: Path, result) -> dict[str, Any]:
        if problem.level != 1:
            return {"skipped": "ncu profiling is implemented for Class 1 tasks only"}
        code = code_path.read_text(encoding="utf-8", errors="replace")
        names = kernel_names(code)[:MAX_KERNELS]
        if not names:
            return {"skipped": "no __global__ kernel found in the candidate"}
        out_dir.mkdir(parents=True, exist_ok=True)
        regex = "^(?:" + "|".join(names) + ")$"
        script = _CHILD.format(root=self.root, out=str(out_dir), ncu=self.ncu, regex=regex, count=len(names),
                               task_id=problem.problem_id, sol=str(code_path), timeout=self.timeout_s)
        env = dict(os.environ)
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        try:
            p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=self.root,
                               env=env, timeout=3 * self.timeout_s)
        except subprocess.TimeoutExpired:
            return {"kernels": names, "error": "ncu collection timed out"}
        status = {"ok": False, "error": f"profiling process exited {p.returncode}: {p.stderr[-800:]}"}
        for line in p.stdout.splitlines():
            if line.startswith("__NCU_JSON__"):
                status = json.loads(line[len("__NCU_JSON__"):])
        if not status.get("ok"):
            return {"kernels": names, "error": status.get("error", "unknown")}

        details = (out_dir / "details.txt").read_text(errors="replace")
        hotspots = ""
        src_rep = out_dir / "source.ncu-rep"
        helper = self.skill_dir / "helpers" / "extract_stall_hotspots.py"
        if src_rep.is_file() and helper.is_file():
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (_ncu_python_dir(self.ncu),
                                                                          os.environ.get("PYTHONPATH", "")) if x))
            subprocess.run([sys.executable, str(helper), "--run-dir", str(out_dir), "--report", str(src_rep),
                            "--tag", "candidate", "--top", "20"], capture_output=True, text=True, env=env,
                           timeout=600)
            hs = out_dir / "analysis" / "stall_hotspots_candidate.txt"
            if hs.is_file():
                hotspots = hs.read_text(errors="replace")

        per_size = "\n".join(
            f"- {s.get('name', '?')}: solution {s.get('kernel_min_ms', '?')} ms, reference {s.get('ref_min_ms', '?')} ms, "
            f"speedup {s.get('speedup', 0):.3f}x" if isinstance(s, dict) else f"- {s}"
            for s in (result.metadata.get("per_size") or []))
        user = (
            f"Task: {problem.problem_id}\n"
            f"Candidate kernels profiled (first launch each, via the task harness, -lineinfo): {', '.join(names)}\n"
            f"Measured by the verifier: overall speedup {result.speedup:.3f}x versus an expert reference "
            f"(geometric mean over sizes).\nPer size:\n{per_size or '- n/a'}\n\n"
            f"## Task specification (every suggestion must respect its rules, e.g. banned libraries and the "
            f"required numerical precision)\n{problem.task_prompt}\n\n"
            f"## Candidate source (`{problem.solution_file}`)\n```cuda\n{code}\n```\n\n"
            f"## ncu --page details (Recipe 1 report: --set full + PM sampling)\n```\n{details[:DETAILS_CHARS]}\n```\n\n"
            f"## Per-source-line stall hotspots (Recipe 2 report, extract_stall_hotspots.py)\n```\n"
            f"{hotspots[:HOTSPOT_CHARS] or 'not available'}\n```\n\n"
            "Follow the skill above: Profile -> Diagnose -> Plan, never guess, cite the metric values. "
            "Reply with ONLY the ranked list of next-step optimization suggestions for this candidate (at most "
            "8 items, most impactful first). Each item: the concrete code change, the metric evidence behind it, "
            "and the expected impact. No preamble and no summary of the report."
        )
        system = ("You are a CUDA performance engineer analysing an Nsight Compute profile with the following "
                  "profiling skill. The target GPU is the one the report was collected on, which may differ "
                  "from the skill's B200 examples.\n\n" + self.skill_prompt)
        try:
            suggestions = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except Exception as exc:
            return {"kernels": names, "error": f"analysis call failed: {type(exc).__name__}: {exc}"}
        (out_dir / "analysis_prompt.md").write_text(user, encoding="utf-8")
        (out_dir / "suggestions.md").write_text(suggestions, encoding="utf-8")
        return {"kernels": names, "suggestions": suggestions.strip()[:SUGGESTION_CHARS],
                "llm": self.llm.last_metadata_dict()}
