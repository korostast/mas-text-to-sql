# v0.1.8 — one reserved finalization repair

This is a narrow runtime follow-up to the v0.1.7 smoke audit, not a grounding or
SQL-policy rewrite. Preserve all prior pilots and the completed v0.1.6 main.

## Unchanged boundaries

Action protocol remains `sequential-author-critic-v2`; system prompt version is
`sequential-action-contract-v3`; public context schema is `public-snapshot-v2`.
Package version is `0.1.8`; allocation policy becomes `closure-reserve-v2`.
Grounding/parser/state, both role system prompts, baseline workflow/single,
GenerationController, actual-usage accounting, transport, executor, dependencies,
configs and all global caps are unchanged. This is not an entailment checker.

## Admission and allocation

A branch now reserves ONE further terminal repair call in addition to the normal
review/explicit-submit chain. Initial admission therefore needs at least four
LLM calls and two SQL slots. New candidates need three further calls, Critic
diagnostics three, Author diagnostics two, and a verdict two. A first submit
attempt requires one further call; its terminal repair does not require another.
The reserved repair executes no SQL unless Author supplies a valid submit action.

For known prompt reservations A/C, growth g=1536 and completion floor
f=min(1024,max_output_tokens), downstream tokens are:

- Candidate or Critic diagnostic: (C+g+f)+(A+2g+f)+(A+3g+f).
- Author diagnostic or verdict: (A+g+f)+(A+2g+f).
- First submit: A+g+f.
- Terminal repair submit: zero additional downstream hold.

Future growth indices cover successive public snapshots and then short private
feedback. Candidate-size rechecks use the SAME candidate-cost function as the
planner. Each current request is still independently tokenized and bounded by
actual remaining tokens/context/output cap. Mutually exclusive branch costs are
not added together; the largest advertised branch is held. Actual usage alone is
charged. No extra budget or extra model-call limit is introduced.

## Terminal state machine

`open -> first -> repair`, with no return to open:

1. Before an Author request, if submit is allowed but candidate/diagnostic are
   not, latch `first` and emit `finalization_started`. Reserve tokens AND one
   LLM slot for a potential repair before issuing that first closing call.
2. A valid explicit submit follows the existing exact-version/fingerprint,
   review, grounded-disagreement and SQLite-success gates. Approval alone is
   never submission. An unused repair hold is not spent or force-consumed.
3. Recoverable protocol/action errors in `first` execute no SQL, preserve the
   candidate/review, latch `repair`, and emit `finalization_repair_started`.
4. Only this repair may release the terminal repair hold. Its action mask permits
   submit only, even if spare tokens/calls could otherwise reopen optional work.
5. A second invalid action emits `finalization_repair_exhausted` and stops with
   that reason. It does not renew the reserve or request a third closing attempt,
   even when global caps have spare capacity.

Normal private error replay outside terminal mode is unchanged. Terminal repair
uses one short user feedback message (error prefix <=240 characters and a fixed
submit-only instruction), NOT the previous rejected assistant response. The
original response is still in the full audit trace. No public source, candidate,
observation or review is deleted. This prevents a large forbidden revision from
being copied into the repair prompt and consuming the very reserve it needs.

Trace `closure_plan` includes `finalization_repair_tokens`,
`finalization_repair_turns` and `finalization_mode` at planning time. Mode can
switch from open immediately after the plan; `finalization_started` records that
transition. Result JSON also includes finalization_mode. The repair plan reports
zero further repair turns/tokens. Generation preflight retains actual prompt
reservation, output cap, holdback and floor.

## Failure limits

This is a bounded opportunity to repair a protocol violation, NOT a guarantee of
a final answer or EX correctness. Prompt growth remains estimated; context limits,
preflight drift beyond the margin, SQL/time/transport/envelope/accounting failures,
a failed initial candidate or another invalid response can still end the episode.
The repair does not add recovery for fatal transport/envelope/accounting failures,
and it is not a reservation for arbitrary earlier Critic errors or failed SQL.
Repeated invalid Critic actions may consume the calls needed for a viable chain;
when no action fits the call mask, runtime stops as insufficient_review_budget.
When no action fits the token allocation it stops as insufficient_closure_budget.
No automatic submission, fallback SQL, raised caps or task-ID special cases exist.
Conservative reservation can disable useful optional work; real cost/EX effects
must be measured, not inferred from synthetic tests.

## Validation

Local scripted-model/toy-SQLite tests cover cap-sized output spending, enlarged
repair prompts, disabled revision/diagnostic, invalid JSON and wrong target,
grounded disagreement, no candidate promotion or third terminal attempt, lower
configured output caps, accounting margins and strict actual usage. They also
check a healthy finalization does not consume an unnecessary repair generation.
Baseline and complete project tests must pass after clean patch application.
Real LLM and BIRD runs are not part of this implementation.

One smoke run on the existing frozen development pilot is the integration check.
Do not tune on per-task gold or launch another main automatically. Preserve
backend/config/caps/evaluator settings and compare coverage, reasons for stops,
terminal events, actual tokens and time against v0.1.7. Improvement is not promised.
