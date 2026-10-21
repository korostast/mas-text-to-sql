# Sequential Author–Critic — v0.1.8

## Scope

`--architecture sequential` implements a fixed-order two-agent team, not an LLM
coordinator, Debate, MoA or Blackboard. Both role agents use the same configured
model and SQLite executor. They have distinct role contexts and one shared budget.
The existing `workflow` and `single` protocols/prompts are unchanged.

- Protocol: `sequential-author-critic-v2`
- Prompt: `sequential-action-contract-v3`
- Context: `public-snapshot-v2`
- Closure allocation: `closure-reserve-v2`
- No new runtime dependencies; Python 3.11+, Linux/POSIX.
- No real model or BIRD benchmark has been run for this release.

## Execution state machine

```text
AUTHOR_INITIAL -> execute candidate -> CRITIC_REVIEW
CRITIC_REVIEW -> optional diagnostic -> CRITIC_REVIEW -> verdict -> AUTHOR_DECIDE
AUTHOR_DECIDE -> optional diagnostic -> AUTHOR_DECIDE
AUTHOR_DECIDE -> changed candidate -> execution -> CRITIC_REVIEW
AUTHOR_DECIDE -> submit current reviewed version -> final execution
  successful final execution -> submitted
  failed final execution -> AUTHOR_DECIDE, if budget permits
```

Only Author owns candidates/submission; only Critic owns review verdicts. Runtime
selects the next role by phase, validates actions, executes tools and charges
resources. Runtime does not judge semantics or choose the best candidate.

The first valid candidate is frozen before any SQL execution feedback or review.
No diagnostic or direct submit can precede it. Candidate changes require a short
public reason and existing sources. SHA256 identifies exact SQL after stripping
outer whitespace only. A change creates a new immutable version and invalidates
its predecessor's approval. Retesting unchanged candidate SQL is rejected.

Each candidate executes before review. Critic can approve or request changes;
it may first execute one diagnostic. Failed candidate execution cannot be
approved. Successful or empty results alone are not semantic correctness proofs.
Author may execute one diagnostic per decision turn, revise, or explicitly submit.
Every revised candidate must be executed and independently reviewed again.

`submit_sql` references the current version/hash, not model-supplied replacement
SQL. Runtime executes the stored SQL again. Approval never automatically submits.
An error does not accept a final answer. Exhaustion never promotes a diagnostic,
a candidate, a failed submit or an approved-but-unsubmitted SQL to `final_sql`.

## Advisory Critic and disagreement

Critic has no veto. After `request_changes`, Author may retain and submit the
current SQL only with `response_to_review` identifying the current review, ALL
of its issue IDs, a short explanation, existing sources and direct evidence grounds. Telemetry preserves
`request_changes` and `submitted_with_disagreement=true`; it is not relabelled as
approval. The runtime verifies structure, verbatim direct quotations, declared
basis and observation provenance/freshness, not argument truth or entailment.
After `approve`, `response_to_review` must be omitted.

A failed final execution is published as an observation with accepted=false.
Author may repair, but repairs still require a new Critic review and consume the
same budget. Repeated failing submissions are discouraged by prompts and remain
bounded by global caps; runtime cannot infer whether a new semantic reason exists.

## Exact action contracts

One JSON object in `message.content`, no code fences or extra prose. Duplicate
keys, unknown fields and fabricated observation envelopes are rejected.

Author:

```json
{"action":"execute_sql","purpose":"candidate","sql":"SELECT ..."}
```

A changed candidate must additionally have:

```json
{"revision":{"reason":"Short change and its basis","sources":["question","review:1"],"basis":"requirement","grounds":[{"source":"question","quote":"EXACT excerpt from the actual task"}]}}
```

Author diagnostic (only after review, once per Author decision turn):

```json
{"action":"execute_sql","purpose":"diagnostic","sql":"SELECT ...","check":"Testable question"}
```

Author submit:

```json
{"action":"submit_sql","candidate_version":1,"sql_fingerprint":"<current SHA256>"}
```

After request_changes, add:

```json
{"response_to_review":{"review_id":"review:1","issue_ids":["I1"],"reason":"Why the candidate should be retained","sources":["question"],"basis":"requirement","grounds":[{"source":"question","quote":"EXACT excerpt from the actual task"}]}}
```

Critic diagnostic (once per candidate review):

```json
{"action":"execute_sql","purpose":"diagnostic","candidate_version":1,"sql_fingerprint":"<current SHA256>","sql":"SELECT ...","check":"Testable question"}
```

Critic verdict:

```json
{"action":"review_sql","candidate_version":1,"sql_fingerprint":"<current SHA256>","verdict":"approve","issues":[]}
```

`request_changes` requires 1..8 issues, each with exactly `id`, `category`,
`message`, `sources`, `basis`, `grounds`. An issue has unique local ID I1..I999, message <=600 chars,
and 1..8 unique source references. Categories: `projection`, `filter`, `join`,
`aggregation`, `ties`, `execution`, `other`. An approval requires no issues.
Revision/rebuttal reasons and diagnostic checks are also <=600 characters.
SQL text is <=100000 characters. References are <=80 characters each.

Only runtime assigns `review:N` and `observation:N`. Valid sources are `question`,
`schema`, nonempty `evidence`, or already published observation/review IDs.
A review is an attributed opinion, not empirical proof. Future/unknown IDs fail.

### Direct grounding gate

Each issue, changed-candidate revision and disagreement carries one `basis` and
1..3 `grounds` objects, each exactly `source` and `quote` (1..240 characters).
Every anchor's source must also occur in the action's `sources`. `review:N` may
remain an attribution/reference but cannot be a ground. Unknown fields, missing
grounds, repeated grounds and v1 issue/revision shapes are rejected, not upgraded.

- `requirement`: at least one exact excerpt from question or nonempty evidence.
- `schema`: at least one exact excerpt from the actual DDL.
- `observed_data`: at least one successful current-version diagnostic.
  The answer candidate's successful output is not a diagnostic.
- `execution`: at least one current failed candidate/submission observation.
  Issues categorized execution must use this basis.

Observation quotes match `json.dumps(result, ensure_ascii=False)`, preserving the
same result-field order and JSON spacing shown in the public snapshot.
Only the JSON result is evidence; SQL/check prose and peer opinions are not.
All observation anchors must target the current candidate version and must not
be truncated. History remains visible/citable, but it is not a fresh empirical
premise. Use a focused diagnostic such as a count or distinct-value test when a
full-set assertion would otherwise rely on a truncated preview.

Author owns its grounds independently; citing only a review no longer admits a
revision. If a premise cannot be supported, keep the candidate and explicitly
disagree, or diagnose if affordable. No mandatory revision/diagnostic is added.

The declared basis is still model-authored. A real quote can be irrelevant and a
model can misclassify a data hypothesis as a requirement. The runtime does NOT
parse natural-language entailment or prove SQL correctness. This is a stronger
auditable provenance gate, not a semantic oracle or a claim of higher EX.

## Contexts and tools

Fresh Author/Critic contexts are created for each task. Initial inputs whitelist
question/evidence/schema; arbitrary task fields such as SQL or EX are not sent.
Every valid call is rebuilt from system instructions, the original task/DDL, and
ONE current public-state packet. Earlier packets and valid assistant replies are
not replayed as a growing conversation: their public contents already live in
versioned state. This is not a new topology or a third agent.

The packet preserves every candidate SQL version exactly once across
`current_candidate` and `candidate_history`. All tool observations remain, with
bounded previews and original IDs. Candidate/submission observations refer to
`candidate_version`, rather than duplicating its SQL. Diagnostic observations
retain their distinct SQL, check and fingerprint. `current_review` is excluded
from historical `reviews`, so every report appears once. `available_sources`
matches the authoritative validator; no old source is silently removed.
`submission_feedback` retains the latest submission's accepted flag, target,
observation ID and rebuttal. Its error/result is in the referenced observation.

Only the latest invalid response/feedback pair can be replayed, privately to its
own role. In terminal repair, replay is instead one bounded feedback message,
without the rejected assistant response; see FINALIZATION_REPAIR.md. It is replaced after another error and cleared on a valid action.
Normal requests have three messages; repair requests have at most five. The
complete audit trace still preserves every actual request/response/action.
Other-role private repair text and reasoning fields never enter a prompt. Logs
may retain valid-envelope reasoning for audit as before, without forwarding it.
A returned packet is detached from state, so serialization cannot mutate records.

Both roles use `database.executor.execute` with the existing read-only worker,
POSIX limits and preview truncation. No silent LIMIT rewrite, external retrieval,
gold access, extra database privileges or extra tools are introduced.

## Shared resources and errors

Defaults remain 8 LLM calls, 8 SQL calls, 32000 total input+output tokens,
4096 output tokens/call, 32768 context tokens and 300 seconds/episode.
Actual repeated prompt tokens, malformed known-usage responses and protocol
repair calls are charged. There is one GenerationController and TokenLedger.
There is no independent full allowance for each agent.

Before admitting an action, runtime reserves ONE terminal repair in the same caps:

- Candidate: at least 3 further LLM calls (Critic, Author, potential repair) and
  2 SQL slots (candidate execution, final execution).
- Critic diagnostic: at least 3 further LLM calls (verdict, Author, repair), and
  2 SQL slots (diagnostic, final).
- Author diagnostic: at least 2 further LLM calls and 2 SQL slots.
- Verdict: at least 2 further LLM calls (Author, repair).
- First final submit: at least 1 further LLM call and 1 SQL slot.
- Terminal repair submit: 1 SQL slot, no further LLM reservation.

Allowed-action packets account for the model call about to be spent; dispatch
checks again after it is charged. An initial budget below 4 LLM calls / 2 SQL calls stops as
`insufficient_review_budget` without contacting the model.

### Token closure allocation

The sequential planner additionally tokenizes the two known role snapshots and
adds the configured preflight margin. These projection requests use the existing
tokenizer endpoint, not an LLM and not SQL. They increase tokenize/solve time but
do not acquire extra generation calls or fabricate usage. Baseline callers do
not opt into this policy.

Let `A`, `C` be the known Author/Critic prompt reservations, `g=1536` the
per-step estimated snapshot growth, and `f=min(1024,max_output_tokens)` the
completion floor. Future branch allocations are:

- new candidate: `(C+g+f) + (A+2g+f) + (A+3g+f)` for review, submit, repair;
- Author diagnostic: `(A+g+f) + (A+2g+f)` for Author, repair;
- Critic diagnostic: `(C+g+f) + (A+2g+f) + (A+3g+f)`;
- review: `(A+g+f) + (A+2g+f)` for Author, repair;
- first submission: `A+g+f` for one bounded terminal repair;
- terminal repair: zero further reserve; no recursive repair allocation.

An optional branch is advertised only when current prompt + floor + its future
allocation fits the remaining global tokens AND calls/tools permit it. Holdback
is the maximum affordable branch, not their sum. Initial Author work must hold
the whole candidate path; a Critic verdict must hold the final Author/repair path.
Author requests also hold the repair allocation until it is actually needed.
After parsing a new candidate, tokenization of a detached state containing its
actual SQL size checks its closure allocation again before any SQL executes.
Placeholder observations in this projection are tokenizer inputs only, never
model feedback or fabricated runtime results.

Generation then caps current output by:
`min(configured output cap, context remainder, global remainder - current prompt reservation - holdback)`.
If this cannot provide `f`, it stops as `insufficient_closure_budget` before a
completion; in particular it does not send a tiny residual request merely to
spend the budget. Every real completion, including protocol repair or malformed
known-usage output, is still charged by actual usage. Unspent holdbacks are not
spending and are recomputed on the next real turn, not carried as a second ledger.

Packets expose allowed actions and closure allocations. When optional work is
disabled, Critic should return a verdict and Author should explicitly submit,
including grounded disagreement when appropriate. A disallowed model action is
not silently converted to submission. Approval never auto-submits.

When Author has only submit available, runtime latches terminal mode. One invalid
closing action gets a charged submit-only repair with bounded error feedback
(no rejected-response replay). A second invalid action stops with
`finalization_repair_exhausted`, even if spare global resources remain. The mode
cannot reopen optional work. Full transitions, formulas, events and failure limits
are specified in `FINALIZATION_REPAIR.md`.

These are enforceable current-call holdbacks based on ESTIMATED future prompts,
not a guarantee of finishing. Actual previews/reviews/private repair text may
outgrow `g`; actual usage can drift beyond the preflight margin; the model can
truncate, fail, ignore the action mask or need more than `f` output tokens.
SQL/time/transport failures remain possible. No budget is raised to conceal them.

Invalid action/JSON/target/source/phase or a resource-reservation rejection keeps
the same active role, executes no SQL, changes no candidate and gives explicit
protocol feedback. A later response is a normal charged call, not a hidden retry.
Completion truncation is recoverable only within these same caps.
Transport/envelope/accounting failures retain v0.1.4 handling and stop precedence,
including unknown usage, actual overruns and strict tokenizer mismatch. There is
no fallback from reasoning to content and no automatic transport retry.

## Modules

| Module | Owns | Must not own |
| --- | --- | --- |
| `agent/sequential/actions.py` | Immutable actions/issues/rebuttals, strict structural parser | Tools, histories, semantic judgement |
| `agent/sequential/grounding.py` | Exact direct anchors, declared basis, provenance/freshness checks | SQL correction, entailment, gold |
| `agent/sequential/closure.py` | Pure branch allocations and token action masks | Transport, actual ledger, SQL |
| `agent/sequential/state.py` | Versions, reviews, references, phase validation, behavior counters | Model calls or SQLite execution |
| `agent/sequential/prompts.py` | Role instructions and public turn packets | Gold, resource ledgers |
| `agent/sequential/roles.py` | Separate Author/Critic histories | Independent budgets or SQL dispatch |
| `agent/sequential/runtime.py` | Fixed order, tools, shared resources, termination, artifacts | SQL generation or correctness oracle |

`agent/runtime.solve` selects the runner. `GenerationController.request` accepts
optional log context and opt-in output floors/holdbacks; `EventLog.scoped` restores metadata after each call.
Default baseline calls preserve their behavior/event shapes. Database and transport semantics
are unchanged. CLI records architecture-specific protocol/prompt versions and the sequential
context/closure versions. Old run artifacts without those fields remain evaluable.

## Artifacts and offline analysis

Existing result fields, BIRD prediction export and investigation schema v1 remain
compatible. `team` adds candidates, reviews, submissions, per-role calls/known
usage, diagnostic counts and final disagreement state. Per-role token values are
KNOWN partials; `complete=false` means they are not full totals.
New `budget_projection`, `closure_plan`, and `candidate_closure_check` events
make forecast token counts, allowed actions and pre-execution rechecks auditable.
They are not model generations or usage charges. Preflight events record the
held downstream tokens and current output floor. Closure plans additionally expose
finalization_repair_tokens/finalization_repair_turns. finalization_started,
finalization_repair_started and finalization_repair_exhausted record the bounded
terminal path; result finalization_mode identifies its latest state.
All generation events carry role/phase/version/context_version. Raw
`critic_review` events use `review_id`; JSONL `id` is reserved for the task and
cannot be overwritten by the event payload. Stored team review objects keep
`id=review:N` for compatibility with source references and the offline evaluator. Tool events distinguish the
executed `sql_fingerprint` from the target `candidate_sql_fingerprint`.

`evaluate` additionally executes reviewed candidate versions against gold OFFLINE,
validates their version/hash identities and writes `review_outcomes` and
`team_quality`. Counts include incorrect approved candidates, requests to change
EX-correct candidates, repairs/regressions and correct finals with disagreement.
These are associations, not proof a critique caused a correction or that each
individual objection was false. Intermediate review scoring adds offline SQL
work, not solve-time model/tool budget. Existing EX/SoftF1/R-VES definitions are
unchanged; zero-final episodes remain in denominators.

## Local validation and real pilot

Tests use synthetic SQLite data and scripted models/local HTTP servers. They do
not estimate real-model accuracy, semantic quality or production latency.

```bash
python -m unittest discover -s tests -v

python -m mas_sql run --architecture sequential \
  --tasks local_data/split/pilot.tasks.json --db-root local_data/databases \
  --config configs/gpt-oss-usage.json --out runs/pilot_v018_sequential_r0 --repeat 0

python -m mas_sql evaluate \
  --tasks local_data/split/pilot.tasks.json --gold local_data/split/pilot.gold.json \
  --db-root local_data/databases --run runs/pilot_v018_sequential_r0 \
  --out runs/pilot_v018_sequential_r0/evaluation.json
```

Use the existing working config/environment for the remote endpoint; no server
reconfiguration is required. Do not overwrite prior runs. Inspect pilot traces
for protocol compliance, call distribution, token use and missing finals before
freezing the main MAS run. Do not tune on individual already-inspected eval IDs.
For comparisons preserve task/database manifests, backend settings, shared caps
and evaluator settings; record the new source hash. No quality gain is promised.


## Grounding policy update

Before requesting a change, Critic is instructed to check exact projection and
every explicit evidence condition, then actual DDL key/nullability constraints,
join multiplicity and the requested unit of counting. Claims about stored
values/duplicates/ties remain hypotheses until supported by task, schema or
observations. A material empirical uncertainty may justify the existing optional
diagnostic; diagnostics and revisions are not mandatory. Author is instructed to
check the premise independently instead of treating a review as fact.

The instructions are now backed by the direct grounding gate above, not a
semantic validator or hardcoded pilot fixes. They do not guarantee correct
reviews, higher EX or a final answer within every budget. Snapshots remove
repeated-state accumulation, but system prompts/task/DDL and all public data
are still charged on every call. Large schemas/previews may still exhaust
resources. Actual accounting, global caps and explicit Author submission remain.
