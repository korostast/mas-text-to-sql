# v0.1.7 — direct grounds and closure allocation

This is a separate post-main experiment built on v0.1.6. Preserve the completed
v0.1.6 main result; do not overwrite it or add corrections keyed to its task IDs.
No server settings, model parameters, global budgets, SQL limits, transport
behavior or dependencies change.

## Version boundary

- Package `0.1.7`.
- Action protocol `sequential-author-critic-v2` (breaking sequential JSON change).
- Prompt `sequential-action-contract-v3`.
- Context `public-snapshot-v2`.
- Closure policy `closure-reserve-v1`.

CLI writes these identities to run/resource/episode/evaluation artifacts. Old
archives remain evaluable. The v1 action schema is not silently accepted as v2.
Workflow and single keep their action parser, prompts and runtime behavior.
GenerationController adds opt-in defaults-preserving reservation parameters.

## Implementation boundary

`grounding.py` is pure validation of short public evidence anchors. `actions.py`
owns strict wire shapes and dataclasses. `state.py` resolves source/target/version
and checks grounds before mutation. `closure.py` is pure allocation arithmetic.
`roles.py` renders role-local contexts. `runtime.py` tokenizes projections,
intersects action masks, checks actual new SQL size and dispatches accepted SQL.
GenerationController enforces current-call holdback/floor while retaining the
single actual TokenLedger. Neither validator nor planner reads gold or chooses
a candidate answer.

The full exact JSON contracts, allocation equations, event fields and failure
semantics are in `SEQUENTIAL_PROTOCOL.md`.

## What becomes mechanically checkable

Review issues, Author revisions and explicit disagreement require a declared
premise type and 1..3 verbatim direct-source anchors. A peer review alone is
insufficient. Anchors must occur in the actual source, match their citation list
and, for observations, be current-version and nontruncated. Data claims need a
successful diagnostic; execution claims need an actual failure.

This does NOT check whether the quotation entails the conclusion, whether the
model declared the correct premise type, or whether the revised SQL is correct.
A focused diagnostic tests a premise, not the gold answer. Semantic quality must
still be measured, including mistaken requirements and unnecessary row removal.

## Closing behavior

Known role snapshots are tokenized for planning. Future branch estimates hold
prompt allowance, growth cushion and completion floors through the required
verdict/explicit-submit path. Optional work is masked if unaffordable. The actual
candidate SQL is rechecked in a detached projection before execution. Projections
do not query SQLite or provide model feedback.

Only actual completion usage is charged. Unused holds are not token spending.
The default output ceiling stays 4096, but a current call can receive a smaller
cap to protect its downstream allocation; calls below the policy floor are
refused. Additional /tokenize requests consume elapsed time, not LLM/SQL slots.
This changes effective allocation and is part of the experimental condition.

No future estimate can guarantee completion under arbitrary model/output/preview
growth or failures. The policy refuses unaffordable work, never auto-submits
approved SQL or falls back to an unsubmitted candidate. Conservative projection
can also reject work that would in hindsight have fit.

## Validation and experiment

Local tests use toy SQLite and scripted/local HTTP responses. They check exact
anchors, wrong/missing quotations, stale/truncated/unsuccessful observations,
independent Author grounding, explicit disagreement, frozen first candidates,
actual usage, malformed responses, output floors, closing masks, candidate-size
rechecks and backwards-compatible baseline behavior.

No real LLM/BIRD run is performed as part of implementation. First run the
existing frozen development pilot, not another main. Compare protocol failures,
unsupported/misclassified grounds, explicit disagreement, finalization coverage,
actual tokens and latency. A short pilot does not establish higher EX or savings.
Agree on any new main/ablation separately; do not silently re-label a tuned main
as untouched test data.