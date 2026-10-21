"""Check direct anchors, declared premise types and explicit limitations."""
import json
import unittest
from typing import Any
from mas_sql.agent.sequential.actions import TeamActionParser
from mas_sql.agent.sequential.grounding import Ground, parse_grounds, validate_grounds
from mas_sql.agent.sequential.state import EpisodeState
from test_sequential import SQL, candidate, issue, review, submit


class GroundingTests(unittest.TestCase):
    """Use synthetic public evidence without benchmark questions or gold."""

    def setUp(self) -> None:
        """Create direct sources and versioned observations."""
        self.texts = {"question": "List every directed edge.", "evidence": "State = 'ACTIVE'",
                      "schema": "CREATE TABLE nodes(id INTEGER PRIMARY KEY, name TEXT);"}
        self.observations: dict[int, dict[str, Any]] = {
            1: {"candidate_version": 1, "purpose": "candidate", "result": {"status": "ok", "rows": [[1], [2]], "preview_truncated": False}},
            2: {"candidate_version": 1, "purpose": "diagnostic", "result": {"status": "ok", "rows": [["ACTIVE"]], "preview_truncated": False}},
            3: {"candidate_version": 0, "purpose": "diagnostic", "result": {"status": "ok", "rows": [[9]]}},
            4: {"candidate_version": 1, "purpose": "candidate", "result": {"status": "error", "error": "no such column: missing"}},
            5: {"candidate_version": 1, "purpose": "diagnostic", "result": {"status": "ok", "rows": [[1]], "preview_truncated": True}},
            6: {"candidate_version": 1, "purpose": "diagnostic", "result": {"status": "error", "error": "timeout"}},
        }

    def check(self, basis: str, source: str, quote: str) -> None:
        """Validate one literal ground at the current version."""
        validate_grounds(basis, (Ground(source, quote),), (source,), self.texts, self.observations, 1)

    def test_exact_question_anchor(self) -> None:
        """A requirement may cite actual task words."""
        self.check("requirement", "question", "every directed edge")

    def test_exact_evidence_anchor(self) -> None:
        """An explicit code definition is available as a direct requirement."""
        self.check("requirement", "evidence", "State = 'ACTIVE'")

    def test_exact_schema_anchor(self) -> None:
        """DDL evidence remains a different declared premise type."""
        self.check("schema", "schema", "id INTEGER PRIMARY KEY")

    def test_successful_diagnostic_anchor(self) -> None:
        """Observed data requires a successful current diagnostic."""
        self.check("observed_data", "observation:2", '"ACTIVE"')

    def test_observation_quote_matches_visible_field_order(self) -> None:
        """A verbatim excerpt spanning result fields is not silently reordered."""
        self.check("observed_data", "observation:2", '"status": "ok", "rows": [["ACTIVE"]]')

    def test_actual_execution_failure_anchor(self) -> None:
        """Execution complaints point to the actual failure."""
        self.check("execution", "observation:4", "no such column")

    def test_paraphrase_is_not_a_quote(self) -> None:
        """A model cannot invent supporting text."""
        with self.assertRaisesRegex(ValueError, "verbatim"):
            self.check("requirement", "question", "Remove redundant rows")

    def test_empty_evidence_has_no_anchor(self) -> None:
        """Absent evidence is not a source merely because the field exists."""
        self.texts["evidence"] = ""
        with self.assertRaises(ValueError):
            self.check("requirement", "evidence", "ACTIVE")

    def test_peer_review_cannot_be_direct_ground(self) -> None:
        """Reference existence is not independent evidence."""
        with self.assertRaisesRegex(ValueError, "review"):
            parse_grounds([{"source": "review:1", "quote": "remove duplicates"}])

    def test_ground_source_must_be_listed(self) -> None:
        """Declared citations and actual anchors must agree."""
        with self.assertRaisesRegex(ValueError, "sources"):
            validate_grounds("requirement", (Ground("question", "edge"),), ("schema",),
                             self.texts, self.observations, 1)

    def test_stale_observation_is_not_a_current_premise(self) -> None:
        """History remains visible but cannot establish a current empirical issue."""
        with self.assertRaisesRegex(ValueError, "current candidate"):
            self.check("observed_data", "observation:3", "9")

    def test_truncated_preview_cannot_block(self) -> None:
        """A focused diagnostic is needed instead of full-set inference."""
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.check("observed_data", "observation:5", "1")

    def test_answer_preview_is_not_a_data_diagnostic(self) -> None:
        """Successful output alone does not establish an empirical hypothesis."""
        with self.assertRaisesRegex(ValueError, "successful diagnostic"):
            self.check("observed_data", "observation:1", "2")

    def test_failed_diagnostic_cannot_establish_data(self) -> None:
        """A diagnostic timeout is not evidence for a data repair."""
        with self.assertRaises(ValueError):
            self.check("observed_data", "observation:6", "timeout")

    def test_successful_execution_is_not_an_execution_failure(self) -> None:
        """Runtime failure grounds require a real failing status."""
        with self.assertRaises(ValueError):
            self.check("execution", "observation:1", "ok")

    def test_wrong_basis_source_rejected(self) -> None:
        """A schema name is not a question requirement."""
        with self.assertRaises(ValueError):
            self.check("requirement", "schema", "PRIMARY KEY")

    def test_unknown_or_malformed_observation_rejected(self) -> None:
        """Unknown references never become evidence."""
        for source in ("observation:999", "observation:no"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.check("observed_data", source, "1")

    def test_ground_bounds_and_uniqueness(self) -> None:
        """Evidence payloads remain short and deterministic."""
        valid = {"source": "question", "quote": "edge"}
        for value in ([], [valid] * 4, [valid, valid], [{"source": "question", "quote": ""}],
                      [{"source": "question", "quote": "x" * 241}], [{"source": "question", "quote": "x", "extra": 1}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_grounds(value)

    def test_v1_issue_and_revision_are_not_silently_upgraded(self) -> None:
        """The versioned contract rejects omitted new grounds."""
        old_issue = {k: v for k, v in issue().items() if k not in ("grounds", "basis")}
        old_revision = candidate(SQL + " WHERE 1=1", ["question"])
        del old_revision["revision"]["grounds"]
        for action, role in ((review(issues=[old_issue]), "critic"), (old_revision, "author")):
            with self.subTest(role=role), self.assertRaises(ValueError):
                TeamActionParser.parse(json.dumps(action), role)

    def test_bad_basis_is_rejected(self) -> None:
        """A peer recommendation is not an evidence basis."""
        value = issue()
        value["basis"] = "critic_said_so"
        with self.assertRaises(ValueError):
            TeamActionParser.parse(json.dumps(review(issues=[value])), "critic")

    def test_ground_checks_do_not_claim_entailment(self) -> None:
        """A real quotation can still accompany a wrong conclusion."""
        state = EpisodeState(True, source_texts={"question": "List item names", "evidence": "name is the label", "schema": "name TEXT"})
        action = TeamActionParser.parse(json.dumps(candidate()), "author")
        state.propose(action, 1)
        state.observe("author", action, {"status": "ok", "rows": [["a"]]}, 1)
        objection = issue()
        objection["message"] = "This quote does not actually entail my proposed edit."
        parsed = TeamActionParser.parse(json.dumps(review(issues=[objection])), "critic")
        state.validate(parsed, "critic")
        self.assertIn("not verified", state.investigation()["grounding_semantics"])

    def test_disagreement_also_needs_direct_ground(self) -> None:
        """Rejecting a reviewer remains an explicit inspectable action."""
        value = submit(disagreement=True)
        value["response_to_review"]["grounds"] = [{"source": "review:1", "quote": "opinion"}]
        with self.assertRaises(ValueError):
            TeamActionParser.parse(json.dumps(value), "author")


if __name__ == "__main__":
    unittest.main()