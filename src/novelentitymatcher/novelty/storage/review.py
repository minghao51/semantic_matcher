"""Lifecycle-aware review and promotion storage for discovery proposals."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from novelentitymatcher.novelty.schemas import (
    NovelClassDiscoveryReport,
    ProposalReviewRecord,
)


class ProposalReviewManager:
    """Persist and update proposal review records for HITL workflows."""

    _ALLOWED_TRANSITIONS = {
        "pending_review": {"approved", "rejected"},
        "approved": {"approved", "promoted"},
        "rejected": {"rejected"},
        "promoted": {"promoted"},
    }

    def __init__(self, storage_path: str | Path = "./proposals/review_records.json"):
        self.storage_path = Path(storage_path)

    def create_records(
        self,
        report: NovelClassDiscoveryReport,
    ) -> list[ProposalReviewRecord]:
        if not report.class_proposals:
            return []

        records = [
            ProposalReviewRecord(
                review_id=str(uuid.uuid4())[:8],
                discovery_id=report.discovery_id,
                proposal_index=index,
                proposal_name=proposal.name,
                proposal=proposal,
                provenance={
                    "discovery_timestamp": report.timestamp.isoformat(),
                    "cluster_ids": list(proposal.source_cluster_ids),
                    "diagnostics": report.diagnostics,
                },
            )
            for index, proposal in enumerate(report.class_proposals.proposed_classes)
        ]
        self.save_records(records)
        return records

    def save_records(self, records: Iterable[ProposalReviewRecord]) -> None:
        payload = [record.model_dump(mode="json") for record in records]
        existing = {record["review_id"]: record for record in self._read_storage()}
        for record in payload:
            existing[record["review_id"]] = record

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(list(existing.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_records(
        self, discovery_id: str | None = None
    ) -> list[ProposalReviewRecord]:
        records = [ProposalReviewRecord(**record) for record in self._read_storage()]
        if discovery_id is None:
            return records
        return [record for record in records if record.discovery_id == discovery_id]

    def update_state(
        self,
        review_id: str,
        state: str,
        *,
        notes: str | None = None,
    ) -> ProposalReviewRecord:
        records = self.list_records()
        updated: ProposalReviewRecord | None = None
        now = datetime.now()

        for index, record in enumerate(records):
            if record.review_id != review_id:
                continue
            self._validate_transition(record.state, state, review_id)
            record.state = state  # type: ignore[assignment]
            record.updated_at = now
            record.notes = notes if notes is not None else record.notes
            if state in {"approved", "rejected"}:
                record.reviewed_at = now
            if state == "promoted":
                record.reviewed_at = record.reviewed_at or now
                record.promoted_at = now
            records[index] = record
            updated = record
            break

        if updated is None:
            raise KeyError(f"Unknown review_id: {review_id}")

        self.save_records(records)
        return updated

    def promote(
        self,
        review_id: str,
        *,
        promoter: Callable[[ProposalReviewRecord], Any] | None = None,
    ) -> ProposalReviewRecord:
        current = next(
            (record for record in self.list_records() if record.review_id == review_id),
            None,
        )
        if current is None:
            raise KeyError(f"Unknown review_id: {review_id}")

        if current.state == "pending_review":
            approved = self.update_state(review_id, "approved")
        else:
            self._validate_transition(current.state, "promoted", review_id)
            approved = current
        if promoter is not None:
            promoter(approved)
        return self.update_state(review_id, "promoted")

    def _read_storage(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Review storage must contain a JSON list")
        return payload

    def _validate_transition(
        self,
        current_state: str,
        new_state: str,
        review_id: str,
    ) -> None:
        allowed = self._ALLOWED_TRANSITIONS.get(current_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid review state transition for {review_id}: "
                f"{current_state} -> {new_state}"
            )
