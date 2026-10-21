# v0.1.6 — grounding and source-preserving contexts

This patch targets v0.1.5. It changes only sequential role prompting/context
presentation and review-event telemetry, plus additive context-version metadata.
No runtime dependencies, database privileges, tool limits or server settings change.

## Identities

- Package: `0.1.6`.
- Action/phase protocol: `sequential-author-critic-v1` (unchanged).
- Prompt: `sequential-action-contract-v2`.
- Context: `public-snapshot-v1`.
- Workflow/single prompts and execution rules: unchanged.

Different context composition is an experimental condition: preserve these
identities and source hash in runs. Do not mix v0.1.5/v0.1.6 episodes into one run.
The evaluator still reads old archives without context_version.

## Grounding, not a semantic oracle

Critic now explicitly prioritizes exact output projection and every condition
in evidence. It must inspect PRIMARY KEY/UNIQUE/nullability and actual join keys
before claiming duplicate entities, and preserve the requested counting unit.
Hypotheses about stored literals, missing values and ties require concrete support
before becoming blocking objections. The existing optional diagnostic is useful
when it distinguishes a material uncertain premise; it is not compulsory.
Author is told to verify the premise and retain SQL with explicit disagreement
when a proposed change is unsupported. No pilot IDs, schema names or gold fixes
are embedded in the prompts. Parser/action schemas are unchanged.

These are instructions to the same model, not a runtime SQL-correctness checker.
Unit tests check wiring/contract text, not semantic compliance by GPT-OSS.

## One packet, not a growing replay

Normal call:

```text
system: this role's instructions
user: original question/evidence/DDL
user: one current public snapshot
```

Only the latest invalid reply and error feedback may be inserted before the
snapshot for that role's next repair call. That pair is replaced after another
error and cleared after a valid action. It never crosses to the peer.

The authoritative state and audit logs retain all valid public history. The
packet includes all sources, not a top-k approximation or an LLM summary:

- `current_candidate` plus `candidate_history`: every SQL version once.
- `current_review` plus historical `reviews`: every report once.
- `observations`: every bounded preview/error and ID. Candidate/submission
  observations refer to candidate_version; diagnostics retain their own SQL/check.
- `submission_feedback`: the latest submission's accepted flag and references,
  including rebuttal when present. Results remain in the cited observation.
- `available_sources`: the same registry used by source validation.
- Remaining resources and allowed actions: unchanged policy.

Raw observations, candidate versions and reports remain unchanged in state and
output artifacts. This context policy changes representation, not SQL semantics,
acceptance, source availability or the shared token accounting. Public claims
can still be wrong; task/result/peer text remains untrusted data.

No automatic submission after approval, last-candidate fallback, extra calls,
independent agent allowances or raised token cap is introduced. Very large
schemas/previews or long episodes can still end without a final submission.
Do not claim measured token savings or higher EX until a real repeat is run.

## Event ID repair

Raw `critic_review` events now expose `review_id`, not top-level `id`.
The CLI assigns persisted JSONL `id` from task identity after event expansion.
Team review objects retain their existing `id=review:N`, so revision references,
submission review_id and offline scoring remain compatible. Old traces are not
rewritten: when reading v0.1.5 JSONL, obtain task identity from the episode/file.

## Local checks

The full suite runs against toy SQLite and scripted/local HTTP model responses.
New checks cover three-version contexts, source availability, single-copy SQL and
review payloads, diagnostic preservation, private/bounded error repair, failed
submission feedback, detached packet data, explicit disagreement, invalidated
approvals, context metadata and JSONL task/review identities.

No real model call or BIRD reexecution has been made for this release. Python
3.11+ remains required; local tests use the installed Python 3.13. Full runtime
source is also checked against Python 3.11 grammar.

## Repeat the pilot only

Use the already working configuration/environment and the same frozen pilot.
No server changes or additional probe are required.

```bash
python -m mas_sql run --architecture sequential \
  --tasks local_data/split/pilot.tasks.json --db-root local_data/databases \
  --config configs/gpt-oss-usage.json --out runs/pilot_v016_sequential_r0 --repeat 0

python -m mas_sql evaluate \
  --tasks local_data/split/pilot.tasks.json --gold local_data/split/pilot.gold.json \
  --db-root local_data/databases --run runs/pilot_v016_sequential_r0 \
  --out runs/pilot_v016_sequential_r0/evaluation.json
```

Substitute actual paths/config names. Do not overwrite old runs or tune main-eval
IDs. Inspect protocol, groundedness of changes, final-answer coverage and actual
usage before freezing the main MAS protocol. Negative quality results remain valid;
this patch does not require the pilot to reach perfect EX.
