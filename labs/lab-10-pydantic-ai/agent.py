"""Lab 10: Type-Safe Production Agents with PydanticAI.

Demonstrates dependency injection, structured Pydantic v2 output validation,
and tool-use guards using PydanticAI (with deterministic offline fallback).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class ComplianceContext:
    """Injected runtime dependencies for the compliance triage agent."""

    jurisdiction: Literal["EU", "US", "APAC"]
    max_risk_score: float = 0.75


class TriageDecision(BaseModel):
    """Structured, schema-validated agent output."""

    summary: str = Field(..., description="Concise summary of the risk assessment.")
    risk_tier: Literal["low", "limited", "high", "unacceptable"]
    requires_human_oversight: bool = Field(
        ..., description="True when EU AI Act Article 14 human-in-the-loop gate is required."
    )
    recommended_controls: list[str] = Field(default_factory=list)


def evaluate_policy_offline(prompt: str, ctx: ComplianceContext) -> TriageDecision:
    """Deterministic fallback when no live LLM API key is configured."""
    lowered = prompt.lower()
    is_high_risk = any(kw in lowered for kw in ("biometric", "clinical", "credit", "hiring", "diagnosis"))
    return TriageDecision(
        summary=f"[{ctx.jurisdiction}] Evaluated request under deterministic policy guardrails.",
        risk_tier="high" if is_high_risk else "limited",
        requires_human_oversight=is_high_risk,
        recommended_controls=[
            "EU AI Act Art. 12 immutable audit logging",
            "GDPR Art. 25 PII redaction before model egress",
            *(["EU AI Act Art. 14 human-in-the-loop approval gate"] if is_high_risk else []),
        ],
    )


def run_pydantic_ai_agent(prompt: str, jurisdiction: Literal["EU", "US", "APAC"] = "EU") -> TriageDecision:
    """Run the PydanticAI agent if configured, otherwise execute deterministic fallback."""
    ctx = ComplianceContext(jurisdiction=jurisdiction)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if api_key:
        try:
            from pydantic_ai import Agent, RunContext

            model_name = "anthropic:claude-3-5-sonnet-latest" if os.getenv("ANTHROPIC_API_KEY") else "openai:gpt-4o-mini"
            agent = Agent(
                model_name,
                deps_type=ComplianceContext,
                output_type=TriageDecision,
                system_prompt=(
                    "You are an Enterprise AI Governance Architect. Classify the AI use case "
                    "and return a structured TriageDecision aligned with EU AI Act and GDPR."
                ),
            )

            @agent.tool
            def get_jurisdiction_threshold(run_ctx: RunContext[ComplianceContext]) -> str:
                return f"Jurisdiction={run_ctx.deps.jurisdiction}, MaxRisk={run_ctx.deps.max_risk_score}"

            result = agent.run_sync(prompt, deps=ctx)
            return result.output
        except Exception as exc:
            print(f"[PydanticAI] Live execution unavailable ({exc}); using deterministic fallback.")

    return evaluate_policy_offline(prompt, ctx)


if __name__ == "__main__":
    sample_prompt = "Deploy a clinical embryo viability scoring model in EU IVF clinics."
    decision = run_pydantic_ai_agent(sample_prompt, jurisdiction="EU")
    print(decision.model_dump_json(indent=2))
