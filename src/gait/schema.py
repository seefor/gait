from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union
import time


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# ----------------------------
# Tokens
# ----------------------------

@dataclass(frozen=True)
class Tokens:
    input_total: Optional[int] = None
    output_total: Optional[int] = None
    estimated: bool = True
    by_role: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------
# Turn
# ----------------------------

@dataclass(frozen=True)
class Turn:
    schema: str
    created_at: str
    user: Dict[str, Any]
    assistant: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    tokens: Tokens = field(default_factory=Tokens)
    visibility: str = "private"

    @staticmethod
    def v0(
        user_text: str,
        assistant_text: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        tokens: Optional[Union[Tokens, Dict[str, Any]]] = None,
        visibility: str = "private",
    ) -> "Turn":

        # normalize tokens input
        if tokens is None:
            tokens_obj = Tokens()
        elif isinstance(tokens, Tokens):
            tokens_obj = tokens
        elif isinstance(tokens, dict):
            tokens_obj = Tokens(
                input_total=tokens.get("input_total"),
                output_total=tokens.get("output_total"),
                estimated=bool(tokens.get("estimated", True)),
                by_role=dict(tokens.get("by_role") or {}),
            )
        else:
            raise TypeError(f"tokens must be Tokens | dict | None, got {type(tokens)}")

        return Turn(
            schema="gait.turn.v0",
            created_at=now_iso(),
            user={"type": "message", "text": user_text},
            assistant={"type": "message", "text": assistant_text},
            context=context or {},
            tools=tools or {},
            model=model or {},
            tokens=tokens_obj,
            visibility=visibility,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)

        # robust tokens serialization
        if isinstance(self.tokens, Tokens):
            d["tokens"] = self.tokens.to_dict()
        elif isinstance(self.tokens, dict):
            d["tokens"] = self.tokens
        else:
            d["tokens"] = Tokens().to_dict()

        return d


# ----------------------------
# Commit
# ----------------------------

@dataclass(frozen=True)
class Commit:
    schema: str
    created_at: str
    parents: List[str]
    turn_ids: List[str]
    snapshot_id: Optional[str]
    branch: str
    kind: str = "auto"
    message: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def v0(
        *,
        parents: List[str],
        turn_ids: List[str],
        branch: str,
        snapshot_id: Optional[str] = None,
        kind: str = "auto",
        message: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Commit":
        return Commit(
            schema="gait.commit.v0",
            created_at=now_iso(),
            parents=parents,
            turn_ids=turn_ids,
            snapshot_id=snapshot_id,
            branch=branch,
            kind=kind,
            message=message,
            meta=meta or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
