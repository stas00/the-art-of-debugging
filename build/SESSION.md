# Session Preferences

## Change approval

1. Always present a concrete proposal before editing source content.
2. Do not apply a proposed correction, addition, or table change until the user explicitly approves it.
3. A request phrased `do N` treats suggestion `N` and any qualifiers in the request as the concrete approved proposal for immediate implementation; do not request a second approval.
4. A request phrased `let's look at N` requests inspection and discussion only; do not edit until the user later approves a proposal.
5. During consistency reviews, produce suggestions rather than silently fixing content. Internal anchor corrections were the only stated exception during the initial review.
6. Never delete or remove any existing content, data, row, note, file, or roadmap entry without first discussing the exact proposed deletion and receiving explicit user approval.
7. Propose deletion only when there is a concrete justification such as invalid, incorrect, harmful, or genuinely redundant content; restructuring, cleanup, uncertainty, or lack of a citation is not sufficient.
8. Before any deletion, explain the reason, scope, consequences, and preservation alternatives so the user can make an informed decision.
9. If restructuring makes existing data difficult to retain in its original location, leave it unchanged and propose a relocation, annotation, or schema adjustment.
10. When replacing or restructuring a table, account for every original row and explicitly flag any row that cannot be represented faithfully.

## Book style

1. Before proposing or writing content, inspect the surrounding section and representative existing sections of the book for established conventions.
2. Follow existing local style for voice, terminology, heading depth, list markers, notes, citations, tables, and source layout rather than introducing a new pattern.
3. When proposing a fix, propose the smallest change that fully corrects the problem. Preserve as much of the original text as possible and avoid rewriting correct surrounding prose merely for polish.
4. Preserve the original author's unique voice, energy, and charm. Correct mistakes without flattening the writing into generic technical prose; broader stylistic rewrites require explicit approval.
5. Prefer the style of the file and nearby section when book-wide usage varies.
6. An explicit user-approved format, such as numbered per-table source lists, overrides a conflicting precedent.
7. If existing style cannot express the content clearly or is inconsistent enough to create a maintenance problem, flag the problem and make a numbered improvement suggestion before changing the style.
8. Reader experience is the first priority when proposing changes. Start with the reader's task: what the reader needs to understand, compare, decide, or do.
9. A technically correct proposal is incomplete if it makes the material less useful, harder to navigate, or harder to apply. Among correct alternatives, prefer the one that best preserves or improves the reader's workflow.
10. For comparison tables, define the comparison question and explain how the reader should use the values before proposing columns, normalization, or vendor-specific qualifications.
11. Make comparison results visible immediately. Precompute useful ratios, deltas, rankings, or normalized values instead of requiring the reader to perform arithmetic.
12. Lead with the reader-facing conclusion or at-a-glance comparison; place provenance, vendor terminology, derivation details, and caveats close by as supporting information rather than making them the primary interface.

## Suggestions report

1. Put a large set of findings in a repository file rather than only in chat.
2. Group findings by severity.
3. Use one flat numerical sequence. A number such as `2` must be sufficient; do not require prefixes such as `HIGH-02`.
4. Make each suggestion independently actionable so suggestions can be applied in any order.
5. When a suggestion is applied, remove it from the report.
6. Never renumber the remaining suggestions after removing an applied item.
7. Include a numerical correction plan ordered by practical priority.

Use the newest `build/consistency-review-*.md` file in the current repository unless the user names a different report.

## Update opportunities report

1. Maintain a separate file for opportunities to extend the book when new accelerators, networking cards, switches, specifications, or standards become available.
2. Do not mix update opportunities with correctness findings.
3. Use the same stable flat-number workflow: make each item independent, remove applied items, and never renumber the remaining items.
4. While researching any topic, add newly discovered, primary-source-supported update opportunities to this file.
5. Propose each update before editing source content.
6. Finish the correctness suggestions before beginning the update queue unless the user explicitly changes that order.

Use the newest `build/update-suggestions-*.md` file in the current repository unless the user names a different report.

## Review scope

1. Check content, logic, numerical arithmetic, units, technical correctness, and internal consistency.
2. Ignore ordinary Markdown whitespace-only differences because rendering already ignores them.
3. Keep Markdown table source vertically aligned. In tables, whitespace matters for maintainer readability even when rendering would be unchanged.
4. Check internal/local links, files, and anchors.
5. Do not perform external-link availability or liveness checks except for newly added links as specified below.

## Companion book sync

1. `pytorch/README.md` in this book and `debug/pytorch.md` in the `ml-engineering` book are the same chapter maintained in two places. Any change to one must be ported to the other in the same pass, in whichever direction the edit started. This is not optional cleanup to schedule later - an unported edit is a divergence that only a cross-book comparison will ever surface, and nothing in either repository points at the other.
2. Prose, code, commands and output port. Relative cross-references do not. The two copies have deliberately disjoint link sets - as of 2026-08-05 this book's copy carries 3 links to its own chapters and the `ml-engineering` copy carries 14 to its own, with no overlap - because each book can only point at chapters it contains. So a link added there to e.g. `../training/dtype.md` has no equivalent here, and this book's links to `../python/README.md` or `../compiled-programs/README.md` have none there. Never "port" a link by path.
3. The same applies to the scripts the two books share. As of 2026-08-05 there are 15 and the expected state is byte-identical: `NicerTrace.py`, `printflock.py`, `see-mem-usage.py`, `underflow_overflow.py`, `torch-distributed-gpu-test.py`, and the `tiny-scripts`/dataset helpers. They sit at different paths - `pytorch/`, `pytorch/code/` and `pytorch/tiny-scripts/` here, `debug/` and `*/tools/` there - so compare by basename, not by path.
4. `README.md` is the one same-named file expected to differ, since each book has its own index. Do not report it.
5. Detect divergence by pairing same-named files by basename and byte-comparing. Excluding `.git/`, `build/` and `trash/`, every pair should be identical; a non-empty result is either an unported edit or a deliberate difference that belongs in the review's `Checks that passed` note.
6. Checking pitfall, 2026-08-05: a first attempt paired the two trees by identical relative path and so compared almost nothing - it matched only the root `README.md`, reported that one expected difference, and looked like a clean pass. A cross-book check that finds nothing is more likely to be mis-wired than correct; confirm it is actually pairing the shared scripts before trusting a clean result.
7. Also exclude the sibling working copies in the parent directory - `ml-engineering1/`, `ml-engineering2/`, `the-art-of-debugging2/`. They are separate snapshots, they will match the pre-edit state and look like divergences, and they must not be edited.
8. Concrete failure, 2026-08-05. The `python -m torch.distributed.run` to `torchrun` sweep was applied across `ml-engineering` only. It broke `torch-distributed-gpu-test.py` out of byte-identity and left this book's copy of the chapter with 11 un-swept command sites plus one in `SKILL.md`, all ported the same day. Note the rule that sweep follows, since it is a rule and not a blind substitution: `torchrun` replaces `python -m torch.distributed.run`, but a site carrying `python -u -m torch.distributed.run` keeps the long form because `torchrun` has nowhere to put `-u`. `pytorch/README.md:2491` is the one such site here.
9. **This file and the companion's copy have diverged badly, and closing that is deferred until work starts on the other book.** Measured 2026-08-06: `ml-engineering` has 21 sections over 308 lines, `the-art-of-debugging` 10 sections over 119. Ten sections exist only in `ml-engineering` and all ten apply to any technical book - `Reader-visible grounding`, `Measured, derived, or assumed`, `Agent-invented content`, `The author's own voice`, `Positional cross-references`, `Internal links and anchors`, `Outdated references and dead artifacts`, `External link rot`, `Glossary sections` and `Unit formatting`. `Product sync map` is the one that does not port, being about tracking hardware products across comparison tables. Six shared sections have also drifted: `Review scope` 12 items against 5, `Cross-vendor hardware tables` 19 against 8 - though much of that gap is specific to hardware tables and may not transfer - `Book style` 16 against 12, `Sources and citations` 10 against 8, `Change approval` 11 against 10, `Suggestions report` 8 against 7. Do not port mechanically: each item was written against a concrete failure in *this* book, and several name files or chapters the other book does not have. Rewrite the reasoning, keep the failure, drop the path.

## Sources and citations

1. Prefer original and primary sources: vendor specifications, official documentation, standards, original papers, and upstream source repositories.
2. When support is missing, recommend a direct link to the relevant specification or primary documentation page.
3. External primary sources may be consulted for factual verification, but they must not be treated as an external-link liveness scan.
4. Always open and confirm the exact target of every new external link before adding it to the repository.
5. The new-link rule is a narrow exception to the no-external-liveness-sweep preference: do not broadly recheck pre-existing external links.
6. Where possible, place citations directly under a numerical table so readers can confirm the displayed values.
7. Keep citations close to the claims or rows they support.
8. Batch command-line liveness checks through `build/check-new-links.sh` and request one reusable approval for that wrapper rather than separate approval for each `curl` invocation.

## Cross-vendor hardware tables

1. Do not force unlike vendor specifications into an artificial normalized schema.
2. Preserve the vendor's documented reporting scope, such as per SM, per CU, per XCD, or per accelerator.
3. Do not derive per-accelerator totals from private local caches unless the derivation is explicitly useful, clearly labeled, and approved.
4. Prefer `not disclosed` or omission over an unsupported estimate.
5. Add a brief explanation of how to compare unlike cache or memory resources, including differences in scope, sharing, semantics, and performance.
6. For new cache-table additions, consider only high-end GPUs.
7. Research current primary specifications, including newly documented parts such as AMD Instinct MI455X, but propose rows before adding them.
8. When one table would mix broadly shared cache capacity with private local resources, prefer separate comparison and vendor-native tables.

## Table ordering and source layout

1. Keep Markdown table source vertically aligned for maintainer readability.
2. Sort table rows by an explicit column and direction.
3. State immediately before the table which column controls the ordering and whether the order is ascending or descending.
4. When adding a row, insert it into the declared order rather than appending it arbitrarily.
5. If a table has no declared sort order, ask the user which order to use and propose an appropriate column before editing it.
6. Do not choose a numerical sort that implies comparability between semantically different vendor specifications.
7. Where practical, use compact source references in the table and place the full live-checked links immediately below it.
8. When a column header is much wider than its body cells, compact it with `<br>` using nearby tables with split headers as the model. Split the header across two or three complete pipe-delimited lines before the separator row, put one segment in each column on every line, append `<br>` to segments that continue, and preserve the full set of `|` delimiters on every header line.
9. After splitting a header, shrink each source column to the minimum width required by its longest header segment or body cell; preserve vertical pipe alignment without redundant padding.
10. Prefer concise, unambiguous abbreviations such as `Uni-dir.` when a full term makes a compact table column unnecessarily wide.
11. Each table has an independent `Ref.` namespace starting at `1`. Ref columns are left-aligned because reference IDs are categorical and may contain multiple comma-separated values. Ref cells and source numbers are plain numbers without brackets or links; the source descriptions below the table contain the actual links.
12. Move explanatory qualifiers out of compact table headers and into nearby prose or notes when the qualifier does not distinguish the displayed values.
13. Format each per-table `Sources:` block as an explicit numbered list whose item numbers match that table's `Ref.` values; do not combine multiple sources into one paragraph.
14. After every table edit, shrink each source column to the minimum width required by its longest header or body cell, while preserving vertical pipe alignment.
15. Keep rendered tables compact to minimize line wrapping on narrow media. Split disproportionately wide headers with `<br>`, use concise labels, and move nonessential detail below the table without sacrificing clarity.

## Source line layout

1. Keep each prose paragraph on one physical source line; do not wrap prose to a fixed line width.
2. Keep each Markdown list item on one physical source line unless it contains nested block content.
3. Preserve intentional blank lines between Markdown blocks.
4. Only code is subject to a line-width limit, which is 119 characters.
5. Wrap code according to the syntax and semantics of its language rather than applying prose-style reflow.
