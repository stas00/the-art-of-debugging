# Session Preferences

## Change approval

0. **NEVER apply anything without an explicit, in-this-turn instruction to apply. This is the single most-violated rule in this collaboration - the user has had to stop the assistant mid-apply repeatedly.** "Anything" means every file without exception: source chapters, code/scripts, `build/SESSION.md`, the review and update reports, everything. Presenting a proposal is not approval. A previous similar edit being approved is not approval. The user answering a clarifying question, refining wording, or asking to `check`/`verify` something is not approval. Reading `next`, `status`, or `re-audit` is not approval. Approval is per-edit and per-turn and looks like `yes`, `apply`, `go`, `go for it`, `do N`, `apply all`. When there is any doubt, present the proposal and STOP - do not edit. Verifying a proposal (compiling, testing, researching) is allowed and encouraged, but the moment it would write to a file, stop and wait.
1. Always present a concrete proposal before editing source content.
2. Do not apply a proposed correction, addition, or table change until the user explicitly approves it.
3. A request phrased `do N` treats suggestion `N` and any qualifiers in the request as the concrete approved proposal for immediate implementation; do not request a second approval.
4. A request phrased `let's look at N` requests inspection and discussion only; do not edit until the user later approves a proposal.
5. During consistency reviews, produce suggestions rather than silently fixing content. Internal anchor corrections were the only stated exception during the initial review.
6. Never delete or remove any existing content, data, row, note, file, or roadmap entry without first discussing the exact proposed deletion and receiving explicit user approval.
7. Propose deletion only when there is a concrete justification such as invalid, incorrect, harmful, or genuinely redundant content; restructuring, cleanup, uncertainty, or lack of a citation is not sufficient.
8. Unverified, rumoured, or unofficial information is not a deletion candidate - it is often exactly what a reader can't get from a vendor page, and removing it because it lacks a citation destroys the most valuable kind of content in the book. Label it as unconfirmed and say what would change if it held, rather than dropping it. "I could not verify this" is a reason to annotate, never a reason to delete.
9. Before any deletion, explain the reason, scope, consequences, and preservation alternatives so the user can make an informed decision.
10. If restructuring makes existing data difficult to retain in its original location, leave it unchanged and propose a relocation, annotation, or schema adjustment.
11. When replacing or restructuring a table, account for every original row and explicitly flag any row that cannot be represented faithfully.

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
13. The book is written in English, so never add text in another language to a chapter. Consulting a non-English source is encouraged when it is the only place a specification is published, but what lands in the chapter is the English translation, not the original string. Do not paste the source text alongside the translation either: a reader who can't read it gains nothing, and it makes the line harder to scan.
14. Translate the meaning rather than transliterating, and normalize the numbers to the book's own [Unit formatting](#unit-formatting) conventions, since a translation is no longer a verbatim quote and the vendor-quote exception no longer applies. Romanize a proper name that has no English form and gloss it once.
15. Say in prose where a figure came from when the source is non-English, so the reader knows the claim is traceable and knows which site to check. Naming the language is useful; reproducing it is not.
16. No politics. The book is about engineering, so keep out trade policy, sanctions, export controls, and national rivalry even when they explain a market fact. Where such a fact matters to a reader's decision, state the fact and stop: "not generally obtainable in this market" is engineering information a reader can act on, while why that is the case is not. This also rules out framing a vendor or a country as a rival, a threat, or a winner.
17. **Never hard-wrap prose. One line per paragraph.** Enforced by `make check-style`, so it is a check rather than something to remember. Tables, code fences and list items keep their own line; a list item's continuation belongs on the item's line. Do not hand-pad tables either - `make fix-tables` re-pads columns and asserts the cell contents are unchanged, and its `--dry-run` exits non-zero, so that is a check too.
18. **Linux only.** This book targets Linux. Validate commands and teach platform behavior for Linux; do not mention macOS, BSD, or other Unix variants unless the author asks. Do not qualify advice with "on Linux … / on macOS …" comparisons - write the Linux form and leave it at that.

## Reader-visible grounding

1. Every number, name, command, or claim that lands in a chapter must be derivable from what the reader can see: the same section, earlier content in the same chapter, or a cross-linked chapter or file in the repository.
2. Never lean on anything that exists only in the assistant's context - benchmark output pasted into chat, a command run while researching, a file read but not quoted, a figure computed in a previous turn. The reader has none of it, and the resulting text looks rigorous while being impossible to check.
3. When a derivation needs a value the chapter does not show, there are exactly two acceptable moves: add the value to the chapter, or restructure the derivation so it uses values that are already there. Silently using the hidden value is the failure mode.
4. This applies with equal force to intermediate steps. A worked example that passes through an unshown quantity is unverifiable even when its final answer happens to be right.
5. Prefer the derivation the reader can redo with the numbers in front of them, even when a hidden value would be more direct.
6. Concrete failure (companion book): a network section derived an elapsed time from an `algbw` figure that lived only in chat-pasted benchmark output, while the section's own table showed only `busbw`. The fix was to recover the time from the table value the reader can see.
7. When quoting or computing from a script's output, adopt the script's own unit definitions rather than assuming them. A script that prints `1GiB = 2**30 Bytes` alongside `1GBps = 10**9 Bytes per second` needs the base conversion in a `GiB / GBps` division.

## Measured, derived, or assumed

1. Every quantity in a chapter is one of three things: measured, derived from measurements, or assumed. The words around it have to say which. `measures 73%` is a claim about an instrument and `works out to 73%` is a claim about arithmetic - they are not interchangeable.
2. The arithmetic being correct is not what is under review here. Each failure below had sound arithmetic and an overstated status, which is exactly why each one survived being written, re-read, and reviewed.
3. Prefer saying a number is unknown over supplying a plausible one. "This has not been measured on that hardware" is worth more to a reader than any figure that would need to be labelled unreliable.
4. A measurement's provenance belongs in the book; the run's setup does not. What tool version, what flags, what hardware, what date - that is provenance. That a package was missing and had to be installed, what had to be retried - that is process, and it is of no use to anyone but whoever obtained the number. Test: would this sentence have been written if the number had come from a datasheet instead of a run? If not, it is process, not provenance.
5. When process detritus does contain something a reader needs, relocate it to the place the reader was when they needed the fact - do not keep it as a measurement footnote.
6. A claim about *mechanism* needs the same discipline as a quantity and is easier to get wrong, because there is no arithmetic to check it against. "and so" is the tell: it asserts a mechanism. Cite the observed mechanism or name none.
7. **Check that two numbers are comparable before comparing them, and if they are not, do not compare them.** A caveat is not a remedy - it labels an invalid inference instead of withdrawing it, and the conclusion still reaches the reader. Ask what differs between the two measurements first. If more than the one variable of interest differs, there is no comparison to make.
8. A worked example illustrating a controlled comparison must not itself be uncontrolled. When a worked example spans more variables than the point needs, suspect that it is concealing something as well as confusing the reader.
9. A spec-rate division gives a bound on time, never the time. Keep constructions like `at wire rate ... would`; as soon as it becomes `crosses in`, a reader takes it as an observation.

## Agent-invented content

1. An assistant working on this book will sometimes produce something genuinely new - a derivation, a conversion, a diagnostic, a way of framing a trade-off - that appears in no source and nowhere earlier in the chapter. That is welcome and worth having. Novelty is not the problem; unreviewed novelty is.
2. So the rule is not "never invent". Forbidding it outright would throw away the useful half and stifle the thing that makes a fresh pass worth running at all. The rule is that anything invented must be surfaced to the user as an invention, named as such, and kept out of the source files until the user has said it makes sense.
3. `Grounded` means one of exactly two things: an authoritative external reference, or content that already appears earlier in the chapter or in a cross-linked file. Anything else is invented - however obvious it feels, and however cleanly it follows from the material around it.
4. Say which kind of invention it is when flagging it, because they fail differently and need different checks. A derivation can be verified on the spot with algebra. A recommendation, threshold, technique, or diagnostic cannot - it needs a source or a measurement, and that is the kind that has caused every problem so far.
5. The failure mode is not a wrong statement, it is a plausible one. Neither looks like a hallucination at the time - that is the whole difficulty.
6. Flag it even when it is almost certainly right. The cost of asking is one sentence; the cost of a confident invention reaching a published chapter is that a reader acts on it.
7. This is stricter than [Change approval](#change-approval), which governs edits the user already has in view. An invention the user did not ask for and cannot see coming has to be named before it is applied, not folded silently into an approved batch.

## The author's own voice

1. This book is written in the first person and much of its value is the author's field experience - case studies, `I have noticed`, cluster anecdotes. That voice is evidence. A reader weighs `I got a whole lot of invalid reports because of it` differently from `this can produce invalid reports`, and rightly so. Which is exactly why it is never available for an assistant to write.
2. Never invent a first-person claim. Do not write `I have seen`, `in my experience`, `this cost me`, `we found`, or any other statement about what the author did, saw, measured, or suffered, unless those words already exist in the repository and you are preserving or relocating them.
3. Concrete failure (companion book, 2026-08-04): a `hint:` explaining two `find` gotchas was given the clause "both of which cost me correct numbers before I understood them". Nothing anywhere in the repository said that. It was manufactured by pattern-matching the register of a genuine anecdote a few paragraphs up. Copying that key produced text that reads more authoritative than anything an assistant is entitled to write, and put a false memory into the author's mouth in the author's own book.
4. The tell is that it feels like good style. A section written in strong first person invites continuation in the same key, and the more convincingly the voice is copied the less likely anyone is to question it. Treat the urge to add colour as the signal to stop.
5. Converting an existing impersonal claim into a personal one is the same error in smaller form. When a general claim genuinely needs weakening, weaken it with `can` or `may` and say what the limitation is - do not anchor it in an experience nobody reported.
6. Write the mechanism instead. A checkable mechanism is more useful than an unverifiable anecdote, because a reader can tell whether it applies to their case.
7. The same restraint covers editorial characterizations pitched in the author's register - calling a workflow `genuinely painful`, a write-up `a goodie`, a vendor practice `a technology deficiency`. Those are judgements the author is entitled to make and an assistant is not.
8. This is a specialization of [Agent-invented content](#agent-invented-content), and stricter in one respect: an invented derivation can at least be checked with algebra and kept once verified, whereas an invented memory cannot be verified by anyone except the author and has no repair other than deletion.

## Positional cross-references

1. `above` and `below` are only safe when the thing referred to is visible from where the reader is standing - the command, output block, table, or list item in the same section. Then leave them alone; a link to the section the reader is already in is noise.
2. Once a reference crosses a heading, replace the bare `above`/`below` with a Markdown link to the target section. The reader who arrived by deep link, search, or a cross-chapter jump has no `above`.
3. This applies to a `####` referring back to material in its parent `###`. The subsection is a landing point in its own right.
4. Prefer naming the target in the link text over keeping the direction word: `as covered in [Tiny models](#...)` rather than `[as covered above](#...)`. Direction words go stale when sections are reordered; names don't.
5. Do not link when the target is already linked in the same sentence or paragraph.
6. Leave comparative `above`/`below` alone - `60W above its TDP`, `below about 1GiB`, `approaches 0 from above` are quantities, not positions.
7. Anchor arithmetic: an em dash in a heading is dropped and the spaces around it each become a hyphen, so `part 1 — do you need` yields `part-1--do-you-need` with a double hyphen. Verify a generated anchor against the file's own existing links rather than trusting a hand-rolled slugifier.

## Internal links and anchors

1. An anchor needs a document to attach to. Link to `pytorch/README.md#anchor`, never to `pytorch#anchor` - the latter resolves to a directory, and a fragment on a directory has nothing to bind to.
2. The `README.md#anchor` form is the dominant convention. GitHub resolves a relative link from a blob page to `/blob/.../dir`, the blob URL of a directory, which does not render the README - so the fragment is dropped.
3. A link to a directory with *no* fragment is fine and needs no change. Only the anchored ones break, so a sweep should target `](path#anchor)` where `path` resolves to a directory.
4. Never let a link checker treat a directory as its `README.md`. A checker that quietly appends it is validating a resolution step that neither GitHub nor any local renderer performs, and it will pass every link in the class above. `build/check-links.py` deliberately does not do that.
5. When generating anchors to check them, model GitHub faithfully: strip backticks and inline links from the heading text, lowercase it, drop every character that is not a word character, space, or hyphen, then replace each remaining space with a hyphen. Repeated identical headings get `-1`, `-2` appended in document order. Do not collapse runs of spaces - see [Positional cross-references](#positional-cross-references) item 7 for the em-dash case where that matters.
6. Sweep the whole repository, not the file just edited. `make check-links-local-fast` is the fast path; `make check-links-local` builds HTML and runs `linkchecker`.
7. A link can resolve perfectly and still be broken. After moving or rewriting any block that other places point at, check the referrers, not the thing you just edited. Grep for every inbound reference to the thing you moved, and for each one confirm the target still contains what the sentence claims it contains.
8. This generalizes past links. Whenever a command lands in a chapter, confirm a reader who arrived at that line cold can obtain and run it - the build is either on the page or one link away. Concrete failure class: dropping `./some-tool` into a chapter because the assistant had it built in its own shell.

## Suggestions report

1. Put a large set of findings in a repository file rather than only in chat.
2. Group findings by severity.
3. Use one flat numerical sequence. A number such as `2` must be sufficient; do not require prefixes such as `HIGH-02`.
4. Make each suggestion independently actionable so suggestions can be applied in any order.
5. When a suggestion is applied, remove it from the report.
6. Do not renumber the remaining suggestions. Removing an applied item leaves a gap, and the gap is correct - a number has to stay a durable identifier so that `do 39` in an old transcript still resolves to the same finding. Record applied numbers in the report's historical sections so the gaps read as history rather than as lost items.
7. Do not renumber without being told to. If the count is what you need, state it in the `Severity summary` line rather than reshuffling the identifiers.
8. Include a numerical correction plan ordered by practical priority.

Use the newest `build/consistency-review-*.md` file in the current repository unless the user names a different report.

## Proposing a fix in chat

When an item needs the author's judgment before it can be applied (anything that is not a mechanical auto-fix), present it in chat using this format, not a one-line summary:

1. **Heading** - the report's item number and its title, so it ties back to the report.
2. **Problem** - what is wrong and why it is wrong. State the contradiction, the failing command, or the incorrect value explicitly. Cite the conflicting source when the problem is a contradiction.
3. **Solution** - the corrected content, shown concretely (the rewritten line, the fixed command, the corrected number). Show enough that the author can judge it without opening the file.
4. **Proposal** - the exact edit you intend to make, phrased as an action the author can approve or decline.

Always link to the specific source with a clickable `file:line` reference (for example `[compiled-programs/README.md:265](../compiled-programs/README.md#L265)`) so the author can jump straight to the line. Link the conflicting source the same way when citing one. Propose one item at a time and wait for approval before editing, unless the author asks for a batch.

When the author approves and you apply a fix, immediately present the next open item in the same reply (same Problem / Solution / Proposal format). Do not stop at a bare "done" and wait for them to say `next`.

When the author asks you to fix or change something in a proposal you already showed, re-show the **whole** corrected block, not just the changed fragment. A diff-style snippet of only the edited part forces the author to mentally reassemble the result and hides formatting damage in the surrounding lines. Re-render the entire item as it would land, every time it changes.

When a proposed block is itself Markdown that contains fenced code, do not wrap the whole proposal in an outer ```` ```markdown ```` fence - the inner fence closes the outer one early and mangles the render. Present such a block as normal message content (its own fences render as intended), or describe it without nesting fences.

The user cannot read long unwrapped lines in chat. Treat this as a hard pre-send gate, not a preference: before sending any reply that shows proposed file text, scan the drafted message and hard-wrap every line inside code fences and block quotes to <=72 characters, rewrapping any that exceed it. This is independent of how the text lands in the file - file content stays one line per paragraph per [Source line layout](#source-line-layout), while the chat *preview* of that same text must be wrapped. Conflating the two (sending the file's one-line-per-paragraph form verbatim into chat) is the recurring failure.

## Update opportunities report

1. Maintain a separate file for opportunities to extend the book when new tools, workflows, or debugging techniques become relevant.
2. Do not mix update opportunities with correctness findings.
3. Use the same stable flat-number workflow: make each item independent, remove applied items, and never renumber the remaining items.
4. While researching any topic, add newly discovered, primary-source-supported update opportunities to this file.
5. Propose each update before editing source content.
6. Finish the correctness suggestions before beginning the update queue unless the user explicitly changes that order.

Use the newest `build/update-suggestions-*.md` file in the current repository unless the user names a different report.

## Review scope

1. Check content, logic, numerical arithmetic, units, technical correctness, and internal consistency.
2. Ignore ordinary Markdown whitespace-only differences because rendering already ignores them. Tables are an exception, as covered by item 3: there whitespace carries readability meaning even though rendering discards it.
3. Keep Markdown table source vertically aligned. In tables, whitespace matters for maintainer readability even when rendering would be unchanged.
4. Check internal/local links, files, and anchors.
5. Do not perform external-link availability or liveness checks except for newly added links as specified below, and except for the periodic redirect sweep described in [External link rot](#external-link-rot) - that one targets silently-relocated URLs rather than link health, and is worth running deliberately.
6. Re-validate every `as of this writing` clause. The phrase means "true when this section was written" rather than "true when the book was last updated" - which future-proofs the sentence for the author but tells the reader only that the claim may be stale, with no way to tell how stale. Treat each as an item to check, never as a disclaimer that excuses staleness.
7. Grep for it case-insensitively. Roughly half the instances are sentence-initial `As of this writing`, so a case-sensitive grep silently misses them.
8. When one checks out, prefer replacing the phrase with the actual date, since the book already does this elsewhere. A dated claim tells the next pass how old it is; `as of this writing` cannot. Use the date the claim was *verified*, not the date `git blame` reports, since blame returns last-touch and any mechanical sweep resets it to today on lines whose claims are years old.
9. Claims about a *dependency's* capabilities or documentation rot fastest - much faster than arithmetic, which mostly stays put. Sweep for `you can't`, `there is no way to`, `not documented`, `not possible`, `they are working on`, and anything described as `new`. `new` rots fastest of all.
10. So when a chapter is edited because a dependency gained a feature, grep the neighbouring files for older statements about that same dependency. The new measurement is what reveals the old claim - nothing else will, because a stale "you can't" reads exactly like a true one.
11. The neighbour can be the next paragraph. Sweeping a chapter for a claim's neighbours is not enough if you do not also verify that the claim was true when written.

## Outdated references and dead artifacts

1. Periodically sweep for content that has quietly aged out rather than become wrong. Two kinds: files nothing points at any more, and references to software or hardware so old that following them wastes the reader's time.
2. Find orphaned files by checking whether each non-`README` file is named anywhere else in the repository. Exclude deliberate scratch / incoming directories rather than reporting them every pass.
3. Prefer inlining a result over parking it in a side file. The book is otherwise largely inline; a side file breaks reading flow.
4. Remember `chapters-md.txt` when a chapter file is added or removed, since it drives the build and the review's own scope count.
5. Sweep for aged technology references with a term list: `pytorch-1.`, `torch==1.`, `cuda-10`, `cuda-11`, `python2`, and old pinned library versions. Most hits will be legitimate - a dated bug report with a linked issue is exactly how such things should be recorded.
6. The ones to act on are those that send a reader somewhere useless. Check the artifact before deleting the advice - the failure mode may still be live even when its stated cause is obsolete.

## Companion book sync

1. `pytorch/README.md` in this book and `debug/pytorch.md` in the `ml-engineering` book are the same chapter maintained in two places. Any change to one must be ported to the other in the same pass, in whichever direction the edit started. This is not optional cleanup to schedule later - an unported edit is a divergence that only a cross-book comparison will ever surface, and nothing in either repository points at the other.
2. Prose, code, commands and output port. Relative cross-references do not. The two copies have deliberately disjoint link sets - as of 2026-08-05 this book's copy carries 3 links to its own chapters and the `ml-engineering` copy carries 14 to its own, with no overlap - because each book can only point at chapters it contains. So a link added there to e.g. `../training/dtype.md` has no equivalent here, and this book's links to `../python/README.md` or `../compiled-programs/README.md` have none there. Never "port" a link by path.
3. The same applies to the scripts the two books share. As of 2026-08-05 there are 15 and the expected state is byte-identical: `NicerTrace.py`, `printflock.py`, `see-mem-usage.py`, `underflow_overflow.py`, `torch-distributed-gpu-test.py`, and the `tiny-scripts`/dataset helpers. They sit at different paths - `pytorch/`, `pytorch/code/` and `pytorch/tiny-scripts/` here, `debug/` and `*/tools/` there - so compare by basename, not by path.
4. `README.md` is the one same-named file expected to differ, since each book has its own index. Do not report it.
5. Detect divergence by pairing same-named files by basename and byte-comparing. Excluding `.git/`, `build/` and `trash/`, every pair should be identical; a non-empty result is either an unported edit or a deliberate difference that belongs in the review's `Checks that passed` note.
6. Checking pitfall, 2026-08-05: a first attempt paired the two trees by identical relative path and so compared almost nothing - it matched only the root `README.md`, reported that one expected difference, and looked like a clean pass. A cross-book check that finds nothing is more likely to be mis-wired than correct; confirm it is actually pairing the shared scripts before trusting a clean result.
7. Also exclude the sibling working copies in the parent directory - `ml-engineering1/`, `ml-engineering2/`, `the-art-of-debugging2/`. They are separate snapshots, they will match the pre-edit state and look like divergences, and they must not be edited.
8. Concrete failure, 2026-08-05. The `python -m torch.distributed.run` to `torchrun` sweep was applied across `ml-engineering` only. It broke `torch-distributed-gpu-test.py` out of byte-identity and left this book's copy of the chapter with 11 un-swept command sites plus one in `SKILL.md`, all ported the same day. Note the rule that sweep follows, since it is a rule and not a blind substitution: `torchrun` replaces `python -m torch.distributed.run`, but a site carrying `python -u -m torch.distributed.run` keeps the long form because `torchrun` has nowhere to put `-u`.
9. **`build/SESSION.md` is shared process rules, adapted per book rather than byte-identical.** Whichever book is actively being worked on is the master for new SESSION changes in that stretch; port them to the companion in the same pass, adapted - rewrite the reasoning, keep the failure, drop paths/`make` targets/sections the other book does not have (e.g. MLE's `Product sync map` and GA-split table rules do not belong here; AoD-only tooling notes do not belong there). A mechanical paste reintroduces wrong paths. Re-synced from MLE into this book on 2026-08-12; subsequent AoD SESSION edits flip the direction.

## Sources and citations

1. Prefer original and primary sources: vendor specifications, official documentation, standards, original papers, and upstream source repositories.
2. When support is missing, recommend a direct link to the relevant specification or primary documentation page.
3. External primary sources may be consulted for factual verification, but they must not be treated as an external-link liveness scan.
4. Always open and confirm the exact target of every new external link before adding it to the repository.
5. The new-link rule is a narrow exception to the no-external-liveness-sweep preference: do not broadly recheck pre-existing external links.
6. Where possible, place citations directly under a numerical table so readers can confirm the displayed values.
7. Keep citations close to the claims or rows they support.
8. Batch command-line liveness checks through `build/check-new-links.sh` and request one reusable approval for that wrapper rather than separate approval for each `curl` invocation.
9. When a vendor or standards-body page appears unreachable, do not conclude the source is unavailable and do not fall back to a secondary source. Many such sites sit behind bot mitigation that rejects a bare HTTP client while serving the same page to a browser user-agent, so a specialized fetch tool can report a false negative. Retry with a normal downloader and a browser user-agent - this is pre-authorized for verifying citations in this repository, and no further approval is needed:

```bash
wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36" URL
```

10. A guessed URL that returns a page is not necessarily the page you wanted - check the title and body before quoting it. If a guess lands on a listing or tag page, follow the site's own links, or read its `/specifications`-style index, to reach the document itself.
11. **Do not propose adding an unmaintained GitHub repository** (archived, README says "not actively maintained" / "research artifact only", last meaningful commit years stale with no successor named). Prefer the live successor when one exists. Repos already cited in the book are left alone unless the author asks; do not sweep them out on this rule alone.
12. **Do not propose adding a GitHub repository with fewer than ~500 stars** as a "notable" tool or framework - stars are a coarse importance filter, not a quality score. Below that bar, mention only if the author asks or the project is already in the book. Existing citations are not swept out on this rule alone.

## External link rot

1. Run `make check-redirects` periodically. It resolves every external URL in `chapters-md.txt` and reports the ones that have moved or died. This is not the broad liveness sweep `Sources and citations` items 3 and 5 rule out - a redirect is a silent correctness problem, not a health check. The old URL still works, nothing looks broken, and the book quietly names a location the project has left. It needs network and takes a while, so run it deliberately rather than in the fast local pass.
2. **Never remove the per-domain rate limiting.** Both `build/check-redirects.py` and `build/check-new-links.sh` serialize requests to a single domain with a few seconds between them, while different domains proceed in parallel. Firing many concurrent requests at one host looks like a scraper and gets the runner throttled or IP-blocked. `--jobs` raises how many *domains* run at once - it is not a way to go faster on one domain, and `--delay` should only ever go up. Budget hours for a full-book run, not minutes. The script prints nothing until it finishes, so an empty log part-way through is not a hang.
3. This applies to external requests only. `build/check-links.py` reads local files and needs no delay.
4. Do not replace a redirect target blindly. The endpoint is sometimes not a new home at all:
   - **Signed CDN URLs.** Hugging Face PDF/EPUB links can resolve to signed, expiring, geo-specific CDN URLs. Pasting those in would break within hours. The script skips known CDN hosts for this reason.
   - **Redirects to a homepage or a generalized index.** The specific document is gone; that is a dead citation needing a new source, not a move.
   - **Redirects to different content.**
   - **A `.git` suffix.** GitHub redirects `repo.git` to `repo` for the web view, but `.git` is correct inside `git clone`.
5. `curl` never sends the `#fragment`, so a reported target never has one. A naive from-to replacement therefore **downgrades every deep link to its page**. Carry the fragment over, and always re-verify the anchor still exists on the new page.
6. Replace longest-first. When one reported URL is a strict prefix of another, replacing the short one first eats the separator of the long one.
7. After any URL sweep, re-extract every changed URL from `git diff` and check each one for a non-200.
8. A `000` result is bot mitigation, not death. Confirm with the browser user-agent recipe in `Sources and citations` item 9 before calling anything dead.
9. Watch the URL extractor itself. Markdown wraps URLs in ways that corrupt a naive regex: a closing paren ends the match, and a URL inside backticks picks the backtick up as its tail.
10. Two moves are invisible to an HTTP check, and `build/check-redirects.py` reports both. A **`<meta http-equiv="refresh">`** is markup rather than a status, so `curl -L` does not follow it and the stale URL answers `200` forever. Chains are why the probe follows more than one hop.
11. When a **rolling alias refreshes to a pinned release**, cite neither end of the chain. Keep the alias and apply the rest of the move - e.g. `/stable/x.html` plus `/2.13/user_guide/y.html` gives `/stable/user_guide/y.html` - then confirm it resolves.
12. A **`403`/`429` carrying a large HTML body is usually a JavaScript browser check**, not throttling and not death. Read the body before concluding anything: `Vercel Security Checkpoint`, `Just a moment...`, `cf-browser-verification` and `Enable JavaScript to continue` are the markers. Do not retry, do not add delay, and do not swap user-agent - the gate wants a JS engine rather than patience.
13. Never quote content from a JS-gated page. The link is fine for a reader with a browser, so it can stay in the book, but its *contents* have not been seen and must not be summarized or cited as verified until someone opens it. See `Measured, derived, or assumed`.
14. Keep `BROWSER_UA` a current and complete browser string. Bot mitigation fingerprints the whole token sequence.
15. When a cited article is withdrawn rather than moved, check the Wayback Machine before concluding the source is unrecoverable - and use the **CDX** endpoint, not the availability API: `web.archive.org/cdx/search/cdx?url=<encoded>&output=json&filter=statuscode:200`. Take the most recent `statuscode:200` capture, and verify the figures are actually in it.
16. A failed query is not evidence of absence. Before writing down a negative result, confirm the check itself ran.
17. **A matching number is not a matching claim.** When hunting a replacement source, finding the figure on a candidate page proves nothing until the surrounding sentence is read. Check the unit, the scope, the generation and the subject, not the digits.
18. **Before replacing a URL, check it is not a substring of another URL in the file.** A withdrawn source often already appears inside an archive link elsewhere, so a bare string replace nests one URL inside another. Anchor the match on the enclosing Markdown - `](<url>)` - or assert the count is 1 before writing. `build/check-links.py` fails on any link whose scheme appears three or more times.

## Cross-vendor hardware tables

This book is not a hardware-spec catalogue; keep this section short. When a rare comparison table does appear:

1. Do not force unlike vendor specifications into an artificial normalized schema.
2. Preserve the vendor's documented reporting scope.
3. Prefer `not disclosed` or omission over an unsupported estimate.
4. Do not choose a numerical sort that implies comparability between semantically different vendor specifications.

(The companion `ml-engineering` book has the full GA-split / availability-table rules; do not import those here.)

## Table ordering and source layout

1. Keep Markdown table source vertically aligned for maintainer readability.
2. Sort table rows by an explicit column and direction when the table is a ranking/comparison.
3. State immediately before such a table which column controls the ordering and whether the order is ascending or descending.
4. When adding a row, insert it into the declared order rather than appending it arbitrarily.
5. If a table has no declared sort order, ask the user which order to use and propose an appropriate column before editing it.
6. Do not choose a numerical sort that implies comparability between semantically different vendor specifications.
7. Where practical, use compact source references in the table and place the full live-checked links immediately below it.
8. When a column header is much wider than its body cells, compact it with `<br>` **inside the single header row**, e.g. `| Platform/<br>example<br>node |`. Never spread a header over several pipe-delimited lines: GFM requires the delimiter row to be the second line of the table, so a multi-line header stops the table from being recognized and GitHub renders the header as literal `|` text with the continuation segments orphaned beneath it.
9. Source column width is set by the longest of the full header string and the body cells. Rendered width is set instead by the longest `<br>`-separated segment, which is why `<br>` narrows the table for the reader even though it lengthens the source line. Optimize for the rendered width; a long source header line is not a problem.
10. Prefer concise, unambiguous abbreviations such as `Uni-dir.` when a full term makes a compact table column unnecessarily wide.
11. Each table has an independent `Ref.` namespace starting at `1` when a Ref column is used. Ref columns are left-aligned because reference IDs are categorical and may contain multiple comma-separated values. Ref cells and source numbers are plain numbers without brackets or links; the source descriptions below the table contain the actual links.
12. Move explanatory qualifiers out of compact table headers and into nearby prose or notes when the qualifier does not distinguish the displayed values.
13. Format each per-table `Sources:` block as an explicit numbered list whose item numbers match that table's `Ref.` values; do not combine multiple sources into one paragraph.
14. After every table edit, shrink each source column to the minimum width required by its longest header or body cell, while preserving vertical pipe alignment.
15. Keep rendered tables compact to minimize line wrapping on narrow media. Compact disproportionately wide headers with `<br>` within the single header row, use concise labels, and move nonessential detail below the table without sacrificing clarity.
16. After editing a table, run `make fix-tables`. It joins multi-line headers into one row, inserts a missing blank line before a table, and re-pads misaligned pipes, then cross-checks the source table count against what `pandoc` renders. It reports `file:line` for everything it fixed, and flags what it cannot fix - such as ragged cell counts, where there is no way to know which cell is missing.

## Glossary sections

1. Keep each glossary list alphabetically sorted, case-insensitively. Insert a new entry in place; never append to the end.
2. Sort per list, not across the section. A chapter may hold several lists, and each is sorted independently.
3. A list whose order is deliberately pedagogical rather than alphabetical may keep that order. Say so in a note, otherwise the next pass will "fix" it.
4. When adding an abbreviation to a chapter, add it to that chapter's glossary in the same edit if the chapter has one. This applies to anything a reader can't expand on sight - and not to vendor names, product model numbers, or terms as widely known as `GPU` or `CPU`.
5. Periodically check both directions: abbreviations used in the body but missing from the glossary, and glossary entries no longer used anywhere in the chapter.

## Source line layout

1. Keep each prose paragraph on one physical source line; do not wrap prose to a fixed line width.
2. Keep each Markdown list item on one physical source line unless it contains nested block content.
3. Preserve intentional blank lines between Markdown blocks.
4. Only code is subject to a line-width limit, which is 119 characters.
5. Wrap code according to the syntax and semantics of its language rather than applying prose-style reflow.

## Unit formatting

1. Write a value tight against its unit, with no separating space: `340Gbps`, `80GiB`, `125TFLOPS`, `700W`. This is the dominant convention across the book.
2. Prefer the `p`-suffixed spelling over the slash spelling: use `TBps` rather than `TB/s`.
3. When quoting a vendor or other external source verbatim, leave the original spelling untouched. The tight rule governs the book's own prose, not quoted material.
4. Unit spacing is a further exception to `Review scope` item 2, and a different one from tables: here the whitespace changes the rendered text, not just the source.
5. Re-check this periodically across the whole book and fix any drift. Spaced forms reappear as new material is added.
6. When fixing units inside a script, change the code and any captured output in the same pass so the two continue to agree. Prose chapters may be fixed independently of scripts. Keep shared scripts byte-identical with the companion book.
7. After changing a unit inside a table, restore vertical pipe alignment and re-shrink the affected columns as required by `Table ordering and source layout`.
