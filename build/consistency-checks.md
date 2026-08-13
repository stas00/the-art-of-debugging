# Consistency Checks (maintainer runbook)

This is a runbook **for a code agent** (or a human maintainer). It is not part of the published book and is not listed in `chapters-md.txt`.

Goal: periodically re-run a set of editorial/QA consistency checks over the book and fix any regressions. Read this file, then perform each check below in order, applying the decision rules and reporting/fixing findings.

All commands assume the repo root as the working directory. Use `rg` (ripgrep) for searching. Paths in this doc are relative to the `the-art-of-debugging` repo root unless noted; the companion book **Machine Learning Engineering** (MLE) is assumed to live at `../ml-engineering`.

Agent instructions:
- Work check by check. For each match, **classify before editing** using the rules given — do not blind-replace.
- Prefer targeted string edits. Never rewrite whole files.
- Leave literal command/API/env values, product-SKU names, and verbatim third-party tool output untouched (see rules).
- When a judgment call is genuinely ambiguous (e.g. a memory *footprint* vs *capacity*), flag it rather than guessing.
- At the end, report a concise summary of what was changed and what was intentionally left.

---

## Check 1 — Byte-unit consistency (GB vs GiB, MB vs MiB, ...)

Find every numeric byte-unit token:

```bash
rg -n '[0-9]\s?[KMGTPkmgtp]i?[Bb]\b' --glob '*.md' --glob '*.py'
```

Classify each hit and normalize per this table:

| Context | Unit | Examples |
| --- | --- | --- |
| On-device memory **capacity** (VRAM, CPU RAM, SRAM, on-chip cache) | binary `KiB/MiB/GiB/TiB` | "80GiB of GPU memory", "pre-allocate 10GiB", RSS/`see_mem_usage` |
| Any quantity **computed via `2**n`** (or reported by a tool that divides by `2**n`/`1024**n`) | binary | `x = torch.ones((n*2**30))`, `see_mem_usage` output, RSS `/2**20` |
| **Bandwidth / throughput** | decimal `GB/s`, `GBps`, `Gbps`, `TBps` | network rates |
| **Disk / storage** capacity & usage, on-disk **file sizes** | decimal `GB/TB` | "2.3TB checkpoint", "1.2GB model file", core file "5GB" |
| **I/O block/file sizes** in an inherently binary context | binary | "block size of 4KiB" |
| Item **counts** (see Check 2) | bare `K/M/B` | "10K samples", "8B params" |

**Never touch** (leave exactly as written):
- Product-SKU names: `A100 80GB`, `H100 80GB HBM3`, etc.
- Literal CLI/API/env values: `dd bs=1G`, `mount -o size=1G`, `systemd-run -p MemoryMax=5G`, `MEMLIMIT=5GB`, `--shm-size=1g`.
- Verbatim third-party tool output: `ls -lh` sizes (`304K`, `5.8M`), `df -h`, `nvidia-smi`, PyTorch OOM messages (already emit `GiB`/`MiB`).

**Author's own scripts:** when a script the author maintains prints a mislabeled unit (e.g. divides by `2**30` but prints `GB`), fix the label in the script too (e.g. `see-mem-usage.py`). After editing any `*.py`, `python3 -m py_compile` it. Keep shared scripts byte-identical with MLE (`../ml-engineering`).

Quick spot-check that the `see_mem_usage` (`[0] mp:`) output has no stale `GB`:

```bash
rg -n 'mp:.* GB\b' --glob '*.md'
```

---

## Check 2 — Bare `k` / `M` qualifiers

```bash
rg -n '[0-9][kKmMgG]\b' --glob '*.md' --glob '*.py'
```

Rule: a bare `K`/`M`/`B` is allowed **only when it counts items** (tokens, parameters, samples, vocab entries, ports, lines, GPUs). If the number denotes **bytes**, give it a real unit per Check 1.

Leave bare: parameter/token counts, dataset names (`openwebtext-10k`), literal command sizes (`bs=1G`), verbatim `ls`/`df` output, and raw numeric values (`64k` ports, `20k` scrollback lines).

---

## Check 3 — Cross-book sync with Machine Learning Engineering (MLE)

Some chapters are shared/overlapping between this book and `../ml-engineering`. Keep their **content** in sync while preserving each book's **conventions**.

Known shared content:
- `pytorch/README.md`  ↔  `../ml-engineering/debug/pytorch.md` (near-identical)
- Shared scripts under `pytorch/` / `pytorch/code/` / `pytorch/tiny-scripts/` ↔ MLE `debug/` and `*/tools/` (byte-identical by basename)
- The "emulating out of resources" memory one-liners in `methodology/README.md` overlap with MLE `debug/*`.

Diff the shared PyTorch chapter:

```bash
diff pytorch/README.md ../ml-engineering/debug/pytorch.md
```

When reviewing the diff, **sync genuine content** (prose wording, numbers, typos, unit fixes) but **do NOT "fix"** these intentional per-book differences:
- **Links** — each book points only at chapters it contains; relative cross-references do not port.
- **Heading/label case** — this book uses sentence-case headings and lowercase labels (`note:`, `important:`, `tldr:`); MLE uses Title Case / `Note:` / `Important:`.
- **Code-fence language tags** — this book tags fences (```` ```python ````/```` ```bash ````); MLE often leaves them bare.

Also apply Check 1 + Check 2 across both trees. Note this book's memory-testing one-liners in `methodology/README.md` allocate via `x 2**30` and display via `/2**20` → those are `GiB`; its `dd`/`tmpfs`/`systemd`/`MEMLIMIT` values are literal and stay as-is.

---

## Check 4 — Internal links & anchors

Every in-repo link target and `#anchor` should resolve. Prefer the fast path that does not need `markdown_it`:

```bash
make check-links-local-fast   # python build/check-links.py over chapters-md.txt
```

Full HTML build + linkchecker:

```bash
make check-links-local
```

GitHub does **not** treat a directory as its `README.md` for an anchored link - `](path#anchor)` where `path` is a directory is broken. `build/check-links.py` flags that class. Fix by correcting the path/anchor (never by deleting the link silently).

---

## Check 5 — External link liveness / redirects

For newly added links only, batch through:

```bash
build/check-new-links.sh URL [URL ...]
```

For a deliberate full-book redirect sweep (slow, needs network):

```bash
make check-redirects
```

Full `linkchecker` including externals:

```bash
make check-links-all
```

Beware **false positives** — GitHub `404`/`429` to bots, JS challenge pages, bot-mitigation `000`s. Only fix a link if it is **genuinely dead**. Prefer an authoritative replacement. See `build/SESSION.md` sections `Sources and citations` and `External link rot`.

---

## Check 6 — Reader-facing programs still compile

```bash
make check-programs
```

Add new reader-copied scripts to `build/check-programs` when they land in a chapter.

---

## Check 7 — Hard-wrapped prose / table padding

```bash
make check-style          # one line per prose paragraph
make fix-tables --dry-run # or: python build/fix-tables.py --dry-run
```

After any table edit: `make fix-tables`.

---

## Reference: unit conventions in one line

Capacity & `2**n`-computed quantities → binary (`GiB`); bandwidth, disk, on-disk file sizes → decimal (`GB`); SKU names / literal command args / verbatim tool output → leave; bare `K/M/B` only for item counts.
