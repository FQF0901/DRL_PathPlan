#!/usr/bin/env python3
"""doc_stats —— 文档/文件「读写调用」统计（只读扫描 + 快照；不删任何东西）。

对给定目录集（默认 ``docs/ .slim/ tools/ tests/ env/specs/ datasets/ runs/``）逐文件输出：

- ``last_write``：最后写时间。**tracked** 文件（``git ls-files``）= ``git log -1
  --format=%ct``（无提交则回退 mtime）；gitignored / untracked = mtime；
- ``refs``：该文件的**仓库相对路径或 basename** 作为**字面量子串**出现在多少个不同
  文本文件里（整仓扫描；排除文件自身、``.venv/``、``.git/``、根级 ``runs*/``、
  ``__pycache__/``、``.pytest_cache/`` 与本工具自身的快照产物——见下"防自指"）。
  注：全路径匹配 ⊂ basename 子串匹配（路径必含 basename），故只搜 basename 即等价；
- ``category``：``keeper`` / ``code-ref`` / ``recent`` / ``stale``（判据见下）。

category 判据（互斥，优先级从上到下）
------------------------------------
1. ``keeper``：显式常驻集（``docs/`` 5 件 + ``.slim/deepwork/p1_v2/README.md``）；
2. ``code-ref``：refs > 0（有任何引用即不判陈旧；引用来自代码或文档都算）；
3. ``stale``：refs == 0 且陈旧度 ≥ ``--days``（默认 30 天）；
4. ``recent``：其余（refs == 0 且未达 stale 判据；含 <14d 与 14d–N 天灰带，
   精确年龄看 ``age_days`` 字段）。

防自指（重要）
------------
快照本身会把全部文件路径写进 JSON 与 ``docs/README.md`` 的"附录：doc_stats 快照"节；
若不排除，跑一次后所有文件都"被引用"、stale 判据失效。故 refs 扫描**恒排除**：

- ``--json`` 输出文件（默认 ``.slim/deepwork/doc_stats.json``）；
- ``docs/README.md`` 快照节（标题行 ``## 附录：doc_stats 快照`` 起、到下一个
  ``## `` 标题或文件尾）内的全部匹配 —— 即快照节必须是该文件最后一节；
- 可用 ``--exclude PREFIX``（可重复）追加排除其它本工具产物（如本轮审计报告）。

已知口径限制（方向说明）：同层 basename 在**同一行内互相重叠**的极罕见情形（如
``a.js`` 与 ``js.map``）按 ``grep -o`` 的 leftmost-longest 只会记其一 —— 只会**少算**
引用（可能把文件误判为更陈旧），对真实 basename 集合（含 450 对包含关系）实测无影响，
且 stale 清单**只列不删**。另：按规格口径只计「路径/basename」**字面量**，dotted
模块名引用（如 ``from tools.diagnostics.forensics_closed_loop import ...``，不含
``.py``）**不计入 refs** —— 抽查 forensics_closed_loop.py：basename 11 处 vs 含模块名
37 处；如需含模块名口径，参考 ``.slim/deepwork/p1_v2/round38/clean4_tools_scan.py``
（basename+stem 双模式，更保守）。

用法::

    tools/venv-python tools/doc_stats.py                      # 摘要到 stdout
    tools/venv-python tools/doc_stats.py --json .slim/deepwork/doc_stats.json
    tools/venv-python tools/doc_stats.py --md                 # 覆盖写 docs/README.md 快照节
    tools/venv-python tools/doc_stats.py --stale-list         # 候选删除清单（只列不删）
    tools/venv-python tools/doc_stats.py --days 14 --roots docs,tools   # 覆盖参数
    tools/venv-python -m tools.doc_stats --selfcheck          # 临时 git 仓库自检
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = ("docs", ".slim", "tools", "tests", "env/specs", "datasets", "runs")
DEFAULT_JSON = ".slim/deepwork/doc_stats.json"
MD_PATH = "docs/README.md"
SNAPSHOT_HEADER = "## 附录：doc_stats 快照"
DEFAULT_DAYS = 30
#: keeper = CLEAN-1 收敛后的 docs/ 常驻 5 件 + P1-v2 工作区索引（不参与 stale 判定）
KEEPERS = frozenset(
    {
        "docs/README.md",
        "docs/architecture_contract.md",
        "docs/chatgpt5p6sol_review.md",
        "docs/deepseekv4p1_argue.md",
        "docs/repo_handbook.md",
        ".slim/deepwork/p1_v2/README.md",
    }
)
#: refs 扫描排除的目录（basename，支持 glob；``runs*`` = 根级 runs 及其变体）
SCAN_EXCLUDE_DIRS = (".git", ".venv", "runs*", "__pycache__", ".pytest_cache")
_TMP_DIR = "/tmp/opencode" if os.path.isdir("/tmp/opencode") else None


# --------------------------------------------------------------------------- #
# git / 扫描基础
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def find_repo_root(start: Path) -> Path | None:
    proc = _git(start, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        return None
    root = proc.stdout.strip()
    return Path(root) if root else None


def tracked_files(repo: Path) -> set[str]:
    proc = _git(repo, "ls-files", "-z")
    if proc.returncode != 0:
        return set()
    return {p for p in proc.stdout.split("\0") if p}


def git_last_write(repo: Path, rel: str) -> int | None:
    """``git log -1 --format=%ct``（committer 时间）；无提交/失败 → None（调用方回退 mtime）。"""
    proc = _git(repo, "log", "-1", "--format=%ct", "--", rel)
    out = proc.stdout.strip()
    if proc.returncode == 0 and out.isdigit():
        return int(out)
    return None


def iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    skip = {"__pycache__", ".pytest_cache"}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            p = Path(base) / f
            if p.is_symlink() and not p.exists():
                continue
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# refs：整仓字面量引用计数
# --------------------------------------------------------------------------- #
def containment_map(patterns: list[str]) -> dict[str, set[str]]:
    """q → {p : p 是 q 的**真子串**}（p ≠ q）。

    用途：``grep -oF`` 多模式取 leftmost-longest，短模式在长模式下会被吞掉；
    对每个命中文本 m 额外记入其全部真子串，即恢复"子串匹配"口径。
    """
    sup: dict[str, set[str]] = {}
    ordered = sorted(set(patterns), key=len)
    for i, p in enumerate(ordered):
        lp = len(p)
        for q in ordered[i + 1:]:
            if len(q) > lp and p in q:
                sup.setdefault(q, set()).add(p)
    return sup


def snapshot_line_range(md_path: Path) -> tuple[int, int] | None:
    """docs/README.md 快照节的 1-based 闭区间行号范围（header → 下一个 ``## `` 前）。"""
    if not md_path.exists():
        return None
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(SNAPSHOT_HEADER):
            start = i
            break
    if start is None:
        return None
    end = len(lines) - 1
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j - 1
            break
    return start + 1, end + 1  # 1-based 闭区间


def _grep_with_patterns(patterns: list[str], head: list[str], tail: list[str]) -> str:
    """``grep <head> -f <pattern_file> <tail>``（patterns 经临时文件传入，-F 字面量）。"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".doc_stats_patterns", delete=False, dir=_TMP_DIR, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(patterns) + "\n")
        pat_file = Path(fh.name)
    try:
        proc = subprocess.run(
            ["grep", *head, "-f", str(pat_file), *tail],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.stdout
    finally:
        pat_file.unlink(missing_ok=True)


def scan_refs(
    repo: Path,
    patterns: list[str],
    sup: dict[str, set[str]],
    *,
    exclude_prefixes: tuple[str, ...] = (),
    always_exclude: tuple[str, ...] = (),
) -> dict[str, set[str]]:
    """对每个 pattern（basename）返回引用它的**去重文件集合**（仓库相对路径）。

    - 单次 ``grep -raonF -f patterns``，``-a`` 二进制按文本（防少算=防假 stale）；
    - 排除目录见 :data:`SCAN_EXCLUDE_DIRS`；
    - 命中文本 m 计入 m + 其全部真子串（恢复子串口径）；
    - 排除 ``always_exclude``（本工具 JSON 输出）与 ``exclude_prefixes``。
    """
    if not patterns:
        return {}
    stdout = _grep_with_patterns(
        patterns,
        ["-raon", "-F", *[f"--exclude-dir={d}" for d in SCAN_EXCLUDE_DIRS]],
        [str(repo)],
    )
    refs: dict[str, set[str]] = {p: set() for p in patterns}
    for line in stdout.splitlines():
        rel, sep, rest = line.partition(":")
        if not sep:
            continue
        ln_str, sep2, match = rest.partition(":")
        if not sep2 or not match:
            continue
        try:
            rel = os.path.relpath(rel, repo)
        except ValueError:
            continue
        skip = False
        for ex in always_exclude + exclude_prefixes:
            if rel == ex or rel.startswith(ex.rstrip("/") + "/"):
                skip = True
                break
        if skip:
            continue
        for pat in {match} | sup.get(match, set()):
            if pat in refs:
                refs[pat].add(rel)
    return refs


# --------------------------------------------------------------------------- #
# 统计主流程
# --------------------------------------------------------------------------- #
def collect_rows(
    repo: Path,
    roots: list[str],
    *,
    days: int = DEFAULT_DAYS,
    keepers: frozenset[str] = KEEPERS,
    json_out: str | None = None,
    md_path: str = MD_PATH,
    exclude_prefixes: tuple[str, ...] = (),
    quiet: bool = False,
) -> tuple[list[dict], dict]:
    tracked = tracked_files(repo)
    now = time.time()

    targets: dict[str, Path] = {}
    for root in roots:
        rp = repo / root
        if not rp.is_dir():
            print(f"[doc_stats] 警告: root 不存在，跳过: {root}", file=sys.stderr)
            continue
        for p in iter_files(rp):
            rel = os.path.relpath(p, repo)
            targets[rel] = p

    # 本工具自身产物（防自指）：JSON 输出 + md 快照节
    json_rel = os.path.normpath(json_out) if json_out else os.path.normpath(DEFAULT_JSON)
    targets.pop(os.path.normpath(DEFAULT_JSON), None)
    if json_rel != os.path.normpath(DEFAULT_JSON):
        targets.pop(json_rel, None)
    md_rel = os.path.normpath(md_path)
    md_lines = snapshot_line_range(repo / md_rel)

    patterns = sorted({Path(rel).name for rel in targets})
    if not quiet:
        print(
            f"[doc_stats] {len(targets)} 个文件 / {len(patterns)} 个 basename 引用模式；扫描 refs …",
            flush=True,
        )

    sup = containment_map(patterns)
    raw_refs = scan_refs(
        repo,
        patterns,
        sup,
        exclude_prefixes=exclude_prefixes,
        # 恒排除默认快照路径（历史遗留产物也会毒化 refs）+ 本次 --json 路径
        always_exclude=tuple({os.path.normpath(DEFAULT_JSON), json_rel}),
    )
    if md_lines is not None:
        start, end = md_lines
        raw_refs = _drop_md_snapshot_refs(repo, md_rel, start, end, patterns, raw_refs, sup)

    rows: list[dict] = []
    for rel, p in sorted(targets.items()):
        try:
            st = p.stat()
        except OSError:
            continue
        lw = git_last_write(repo, rel) if rel in tracked else None
        if lw is None:
            lw = int(st.st_mtime)
        age_days = max(0.0, (now - lw) / 86400.0)
        refs_n = len(raw_refs.get(Path(rel).name, set()) - {os.path.normpath(rel)})
        if rel in keepers:
            cat = "keeper"
        elif refs_n > 0:
            cat = "code-ref"
        elif age_days >= days:
            cat = "stale"
        else:
            cat = "recent"
        root = _root_of(rel, roots)
        rows.append(
            {
                "path": rel,
                "root": root,
                "refs": refs_n,
                "last_write": lw,
                "age_days": round(age_days, 1),
                "category": cat,
                "tracked": rel in tracked,
                "bytes": st.st_size,
            }
        )

    counts: dict[str, dict[str, int]] = {}
    for root in roots:
        sub = [r for r in rows if r["root"] == root]
        if not sub and not (repo / root).is_dir():
            continue
        counts[root] = {
            "n": len(sub),
            **{c: sum(1 for r in sub if r["category"] == c) for c in ("keeper", "code-ref", "recent", "stale")},
            "bytes": sum(r["bytes"] for r in sub),
        }
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(repo),
        "roots": roots,
        "days": days,
        "n_files": len(rows),
        "counts": counts,
        "refs_semantics": (
            "refs = 路径或 basename 字面量在整仓文本的去重引用文件数；"
            "排除自身/.venv/.git/runs*/__pycache__/.pytest_cache/本快照产物（JSON + docs/README.md 快照节）"
        ),
        "category_rules": "keeper > code-ref(refs>0) > stale(refs==0 且 age>=days) > recent(其余)",
    }
    return rows, meta


def _drop_md_snapshot_refs(
    repo: Path,
    md_rel: str,
    start: int,
    end: int,
    patterns: list[str],
    raw_refs: dict[str, set[str]],
    sup: dict[str, set[str]],
) -> dict[str, set[str]]:
    """把 md 快照节内的引用从计数里去掉（重扫一次 md 文件拿行号）。

    grep 合并输出后无法回溯行号，故对 ``md_rel`` 单文件重跑一次带 ``-n`` 的 grep：
    - pattern 在快照节**内**出现且**节外**未出现 → 从该 pattern 的引用集合删去 md_rel；
    - 节外也出现（真实引用，如 README 正文的规则行）→ 保留。
    """
    md_abs = repo / md_rel
    stdout = _grep_with_patterns(patterns, ["-aon", "-F"], [str(md_abs)])
    in_section: set[str] = set()
    outside: set[str] = set()
    for line in stdout.splitlines():
        ln_str, sep, match = line.partition(":")
        if not sep or not ln_str.isdigit():
            continue
        credits = {match} | sup.get(match, set())
        (in_section if start <= int(ln_str) <= end else outside).update(credits)
    only_section = in_section - outside
    if not only_section:
        return raw_refs
    out: dict[str, set[str]] = {}
    for pat, files in raw_refs.items():
        out[pat] = {f for f in files if os.path.normpath(f) != md_rel} if pat in only_section else files
    return out


def _root_of(rel: str, roots: list[str]) -> str:
    best = ""
    for root in roots:
        if rel == root or rel.startswith(root.rstrip("/") + "/"):
            if len(root) > len(best):
                best = root
    return best or rel.split("/")[0]


# --------------------------------------------------------------------------- #
# 输出：摘要 / JSON / md 快照 / stale 清单
# --------------------------------------------------------------------------- #
def fmt_ts(ts: float) -> str:
    """北京时间（UTC+8，不受容器 TZ 影响）。"""
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts + 8 * 3600))


def fmt_size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} kB"
    return f"{n} B"


def print_summary(rows: list[dict], meta: dict, top: int = 10) -> None:
    print(f"[doc_stats] {meta['n_files']} 文件；days={meta['days']}")
    for root, c in meta["counts"].items():
        print(
            f"  {root:12s} n={c['n']:<6d} keeper={c['keeper']:<4d} code-ref={c['code-ref']:<6d} "
            f"recent={c['recent']:<6d} stale={c['stale']:<6d} {fmt_size(c['bytes'])}"
        )
    stale = sorted((r for r in rows if r["category"] == "stale"), key=lambda r: -r["age_days"])
    print(f"[doc_stats] stale 候选 {len(stale)} 项（refs==0 且 age>={meta['days']}d；只列不删）")
    for r in stale[:top]:
        print(f"  {r['age_days']:7.1f}d | {fmt_ts(r['last_write'])} | {r['path']}")
    if len(stale) > top:
        print(f"  …（其余 {len(stale) - top} 项见 --stale-list / --json）")


def write_json(path: Path, rows: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"[doc_stats] wrote {path}")


def render_md_section(rows: list[dict], meta: dict, top: int = 20) -> str:
    lines = [SNAPSHOT_HEADER, ""]
    lines.append(
        f"> 生成: {fmt_ts(time.time())}（CST）｜生成器: `tools/venv-python tools/doc_stats.py "
        f"--json .slim/deepwork/doc_stats.json --md`｜口径: refs = 路径/basename 字面量在整仓"
        f"（排除 .venv/.git/runs*/__pycache__/.pytest_cache 与本快照自身）的去重引用文件数；"
        f"category = keeper > code-ref(refs>0) > stale(refs==0 且陈旧≥{meta['days']}d) > recent。"
    )
    lines.append(f"> 完整逐文件明细（path/refs/last_write/age/category）: `.slim/deepwork/doc_stats.json`"
                 f"（gitignored）；重跑方法见工具 docstring。")
    lines.append("")
    lines.append("| root | 文件 | keeper | code-ref | recent | stale | 体积 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for root, c in meta["counts"].items():
        lines.append(
            f"| `{root}/` | {c['n']} | {c['keeper']} | {c['code-ref']} | {c['recent']} | {c['stale']} | {fmt_size(c['bytes'])} |"
        )
    stale = sorted((r for r in rows if r["category"] == "stale"), key=lambda r: -r["age_days"])
    lines.append("")
    lines.append(
        f"stale 候选（refs==0 且陈旧 ≥{meta['days']} 天，共 {len(stale)} 项；**只列不删**，全量 `--stale-list`）Top-{top}:"
    )
    lines.append("")
    if stale:
        lines.append("| 陈旧(天) | last_write | 文件 |")
        lines.append("| --- | --- | --- |")
        for r in stale[:top]:
            lines.append(f"| {r['age_days']:.1f} | {fmt_ts(r['last_write'])} | `{r['path']}` |")
        if len(stale) > top:
            lines.append(f"| … | | 其余 {len(stale) - top} 项见 `.slim/deepwork/doc_stats.json` |")
    else:
        lines.append("（无）")
    return "\n".join(lines) + "\n"


def write_md_snapshot(path: Path, rows: list[dict], meta: dict) -> None:
    section = render_md_section(rows, meta)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    start = end = None
    for i, line in enumerate(lines):
        if line.startswith(SNAPSHOT_HEADER):
            start = i
            break
    if start is not None:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        new_lines = lines[:start] + [section] + lines[end:]
        new_text = "".join(new_lines)
        if not new_text.endswith("\n"):
            new_text += "\n"
    else:
        new_text = text
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        if new_text and not new_text.endswith("\n\n"):
            new_text += "\n"
        new_text += section
    path.write_text(new_text, encoding="utf-8")
    print(f"[doc_stats] md 快照{'更新' if start is not None else '写入'}: {path}「{SNAPSHOT_HEADER}」")


def print_stale_list(rows: list[dict], days: int) -> None:
    stale = sorted((r for r in rows if r["category"] == "stale"), key=lambda r: -r["age_days"])
    print(f"# doc_stats stale 候选（refs==0 且陈旧 ≥{days} 天；只列不删）共 {len(stale)} 项")
    print("# 陈旧(天) | last_write(CST) | path")
    for r in stale:
        print(f"{r['age_days']:9.1f} | {fmt_ts(r['last_write'])} | {r['path']}")


# --------------------------------------------------------------------------- #
# selfcheck：临时 git 仓库造 tracked/untracked/零引用样例，断言分类
# --------------------------------------------------------------------------- #
def selfcheck() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="doc_stats_selfcheck_", dir=_TMP_DIR))
    ref = tmp / "ref"
    ref.mkdir()
    now = time.time()
    old_ts = int(now - 60 * 86400)

    (ref / "source.py").write_text("# 引用样例：used_old.py\n", encoding="utf-8")
    (ref / "used_old.py").write_text("# 被引用（tracked 且旧）\n", encoding="utf-8")
    (ref / "unused_old.py").write_text("# 零引用（tracked 且旧）\n", encoding="utf-8")
    (ref / "self_ref.py").write_text("# self_ref.py 自引用不计\n", encoding="utf-8")
    (ref / "keeper.md").write_text("# keeper 样例\n", encoding="utf-8")
    (ref / "unused_new.txt").write_text("零引用但新\n", encoding="utf-8")
    os.utime(ref / "keeper.md", (old_ts, old_ts))

    _git(tmp, "init", "-q")
    _git(tmp, "add", "ref/source.py", "ref/used_old.py", "ref/unused_old.py")
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": f"{old_ts} +0800",
        "GIT_COMMITTER_DATE": f"{old_ts} +0800",
    }
    proc = subprocess.run(
        ["git", "-C", str(tmp), "-c", "user.email=selfcheck@local", "-c", "user.name=selfcheck",
         "commit", "-q", "-m", "init"],
        capture_output=True, text=True, env=env, check=False,
    )
    if proc.returncode != 0:
        print(f"[selfcheck] git commit 失败: {proc.stderr}", file=sys.stderr)
        return 1

    json_out = "snap/doc_stats.json"
    md_path = "README.md"
    failures: list[str] = []

    def run() -> tuple[list[dict], dict]:
        return collect_rows(
            tmp,
            ["ref"],
            days=30,
            keepers=frozenset({"ref/keeper.md"}),
            json_out=json_out,
            md_path=md_path,
            quiet=True,
        )

    rows, meta = run()
    by_path = {r["path"]: r for r in rows}
    expect = {
        "ref/used_old.py": ("code-ref", 1),   # tracked 旧 + 被引用 → code-ref
        "ref/unused_old.py": ("stale", 0),    # tracked 旧 + 零引用 → stale（证明 last_write 走 git log）
        "ref/self_ref.py": ("recent", 0),     # 自引用不计 → refs 0；新 → recent
        "ref/keeper.md": ("keeper", 0),       # 显式 keeper（旧且零引用也不 stale）
        "ref/unused_new.txt": ("recent", 0),  # untracked 新 → recent
        "ref/source.py": ("stale", 0),        # tracked 旧 + 零引用（引用别人不等于被引用）→ stale
    }
    for path, (cat, refs) in expect.items():
        row = by_path.get(path)
        if row is None:
            failures.append(f"{path}: 行缺失")
            continue
        if row["category"] != cat:
            failures.append(f"{path}: category={row['category']} != {cat}")
        if row["refs"] != refs:
            failures.append(f"{path}: refs={row['refs']} != {refs}")
    age = by_path.get("ref/unused_old.py", {}).get("age_days", 0)
    if not 55 <= age <= 65:
        failures.append(f"ref/unused_old.py: age_days={age}（期望 ~60，证明 tracked 走 git log 而非 mtime）")
    if by_path.get("ref/unused_new.txt", {}).get("tracked"):
        failures.append("ref/unused_new.txt: 不应为 tracked")

    # 防自指：写快照（JSON + md 快照节）后重跑，refs 必须不变
    write_json(tmp / json_out, rows, meta)
    write_md_snapshot(tmp / md_path, rows, meta)
    rows2, _ = run()
    by_path2 = {r["path"]: r for r in rows2}
    for path in expect:
        r1, r2 = by_path[path]["refs"], by_path2.get(path, {}).get("refs", -1)
        if r1 != r2:
            failures.append(f"{path}: 快照写入后 refs 变化 {r1} -> {r2}（防自指失效）")
    if by_path2.get("ref/unused_old.py", {}).get("category") != "stale":
        failures.append("ref/unused_old.py: 快照写入后不再 stale（md 快照节未排除）")
    # 默认 JSON 路径恒排除：即使本轮 --json 指向别处，遗留的默认快照也不能毒化 refs
    default_snap = tmp / ".slim/deepwork/doc_stats.json"
    default_snap.parent.mkdir(parents=True)
    default_snap.write_text('{"rows":[{"path":"ref/unused_old.py"}]}\n', encoding="utf-8")
    rows3, _ = run()
    by_path3 = {r["path"]: r for r in rows3}
    if by_path3.get("ref/unused_old.py", {}).get("refs", -1) != 0:
        failures.append("默认 .slim/deepwork/doc_stats.json 未被排除（refs 被毒化）")

    if failures:
        print("[selfcheck] FAIL：\n  " + "\n  ".join(failures))
        print(f"[selfcheck] 现场保留: {tmp}")
        return 1
    print("[selfcheck] OK：tracked/untracked/零引用/自引用/keeper/防自指 6 组断言全过")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="doc_stats.py", description="文档/文件读写调用统计（只读 + 快照；不删任何东西）"
    )
    ap.add_argument("--repo", default=None, help="仓库根（缺省 = 从 cwd 探测 git toplevel）")
    ap.add_argument(
        "--roots",
        default=",".join(DEFAULT_ROOTS),
        help=f"逗号分隔的目录集（仓库相对）；缺省: {','.join(DEFAULT_ROOTS)}",
    )
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="stale 阈值（天；默认 30）")
    ap.add_argument(
        "--json",
        nargs="?",
        const=DEFAULT_JSON,
        default=None,
        help=f"输出逐文件 JSON（可省略路径，默认 {DEFAULT_JSON}）",
    )
    ap.add_argument("--md", action="store_true", help=f"覆盖写 {MD_PATH} 的「{SNAPSHOT_HEADER}」节")
    ap.add_argument("--md-path", default=MD_PATH, help=f"md 快照目标（默认 {MD_PATH}）")
    ap.add_argument("--stale-list", action="store_true", help="打印候选删除清单（只列不删）")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="refs 扫描追加排除的路径前缀（可重复；用于排除本工具审计报告等自身产物）",
    )
    ap.add_argument("--selfcheck", action="store_true", help="临时 git 仓库自检后退出")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selfcheck:
        return selfcheck()

    repo = Path(args.repo).resolve() if args.repo else (find_repo_root(Path.cwd()) or ROOT)
    roots = [s.strip() for s in args.roots.split(",") if s.strip()]
    rows, meta = collect_rows(
        repo,
        roots,
        days=args.days,
        json_out=args.json,
        md_path=args.md_path,
        exclude_prefixes=tuple(args.exclude),
    )
    print_summary(rows, meta)
    if args.stale_list:
        print_stale_list(rows, args.days)
    if args.json:
        write_json(repo / args.json, rows, meta)
    if args.md:
        write_md_snapshot(repo / args.md_path, rows, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
