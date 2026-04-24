#!/usr/bin/env python3
"""
LLM-Enhanced Chat Interface

Purpose:
- Keep deterministic engine as source of truth.
- Let LLM explain results, not invent conclusions.
- User asks questions about experiment outputs.
- LLM uses JSON summaries/stability/results as grounded evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from chat import ExperimentTools, ChatInterface

load_dotenv("venv/.env")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SYSTEM_PROMPT = """
You are a statistical learning tutor explaining deterministic experiment results.

Core rule:
The deterministic research engine is the source of truth. Your job is to explain it clearly.

Strict rules:
- Use ONLY the provided experiment data.
- Do NOT invent trading implications.
- Do NOT mention transaction costs, slippage, market impact, profitability, live trading, or execution unless explicitly present in the data.
- Do NOT override the deterministic decision/classification.
- Do NOT say a signal is strong unless the provided strength/classification says so.
- If data is missing, say it is missing.
- Prefer concrete numbers from the provided data.
- Keep answers grounded, short, and specific.

CLARITY RULE (VERY IMPORTANT):
- Avoid vague phrases like "does not generalize well", "weak performance", or "unclear behavior".
- Instead, explain the exact reason using metrics (e.g., "significance does not survive the train/test split").

STATISTICAL INTERPRETATION RULES:
- If p-values are significant → say the signal exists in-sample.
- If R² is low → say explanatory power is weak.
- Do NOT suggest that stronger p-values are needed if they are already significant.
- Focus on stability when explaining REFINE decisions.

MANDATORY STABILITY ANALYSIS:
If stability data is present, you MUST mention:
- significance_survival_rate (exact percentage)
- direction_consistent (True/False)

CRITICAL RULE:
If significance_survival_rate == 0:
- This is the PRIMARY reason for REFINE over PROMOTE.
- You MUST explicitly say:
  "the signal is statistically significant in-sample, but does not remain significant after the train/test split"

FORBIDDEN CONTENT:
- Do NOT mention "bucket support" or any bucket-related concepts unless explicitly present in the data.
- Do NOT introduce concepts not tested in the experiment.
"""


class LLMChatInterface:
    """Chat interface with grounded LLM explanations."""

    def __init__(self) -> None:
        self.tools = ExperimentTools()
        self.base_chat = ChatInterface()
        self.conversation_history: List[Dict[str, str]] = []
        self.active_context = {
            "symbols": [],
            "family": None
        }

        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=api_key) if api_key else None
            if self.openai_client:
                print("✅ OpenAI client initialized")
            else:
                print("⚠️ OPENAI_API_KEY not found. Falling back to rule-based answers.")
        else:
            self.openai_client = None
            print("⚠️ OpenAI package not installed. Falling back to rule-based answers.")

    def process_question_with_llm(self, question: str) -> str:
        self.conversation_history.append({"role": "user", "content": question})

        deterministic_answer = self.base_chat.process_question(question)

        if not self._needs_llm_explanation(question, deterministic_answer):
            self.conversation_history.append({"role": "assistant", "content": deterministic_answer})
            return deterministic_answer

        response = self._get_llm_explanation(question, deterministic_answer)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _needs_llm_explanation(self, question: str, deterministic_answer: str) -> bool:
        q = question.lower()
        return (
            "why" in q
            or "explain" in q
            or "understand" in q
            or "what does" in q
            or "didn't understand" in deterministic_answer.lower()
        )

    def _get_llm_explanation(self, question: str, deterministic_answer: str) -> str:
        if not self.openai_client:
            return self._get_rule_based_explanation(question, deterministic_answer)

        evidence = self._collect_grounded_evidence(question)
        prompt = self._build_prompt(question, deterministic_answer, evidence)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=650,
            )

            content = response.choices[0].message.content or ""
            return f"🤖 Grounded Analysis:\n\n{content}"

        except Exception as e:
            return f"⚠️ OpenAI API error: {e}\n\n{self._get_rule_based_explanation(question, deterministic_answer)}"

    def _collect_grounded_evidence(self, question: str) -> Dict[str, Any]:
        available = self.tools.list_available_data()
        q = question.lower()

        evidence: Dict[str, Any] = {
            "question": question,
            "matched_symbols": [],
            "symbols": {},
        }

        symbols = list(available.get("symbols", {}).keys())
        matched_symbols = [s for s in symbols if s.lower() in q]

        # If no symbol is mentioned, keep evidence compact.
        if not matched_symbols:
            evidence["available_symbols"] = symbols[:20]
            return evidence

        evidence["matched_symbols"] = matched_symbols
        
        # Update active context when symbols/family are matched
        self.active_context["symbols"] = matched_symbols
        
        # Extract family from question to update context
        for family in available.get("families", []):
            if family.lower() in q:
                self.active_context["family"] = family
                break

        for symbol in matched_symbols:
            families = available["symbols"].get(symbol, [])
            matched_families = self._match_families(q, families)

            # Fallback: if user mentions only symbol, include all family summaries.
            if not matched_families:
                matched_families = families[:5]

            evidence["symbols"][symbol] = {}

            for family in matched_families:
                base_family = self._normalize_family_name(family)

                summary = self.tools.get_summary(symbol, base_family)
                stability = self.tools.get_stability(symbol, base_family)
                experiments = self._load_experiment_details(symbol, base_family)

                evidence["symbols"][symbol][base_family] = {
                    "family_summary": summary,
                    "stability": stability,
                    "experiments": experiments,
                }

        return evidence

    def _match_families(self, question_lower: str, families: List[str]) -> List[str]:
        matches = []

        for family in families:
            fam = str(family).lower()
            normalized = fam.replace("_", " ")

            if fam in question_lower or normalized in question_lower:
                matches.append(family)
                continue

            # Looser keyword matching
            keywords = [part for part in fam.split("_") if len(part) > 2]
            if keywords and all(k in question_lower for k in keywords[:2]):
                matches.append(family)

        return matches

    def _normalize_family_name(self, family: str) -> str:
        fam = str(family).lower()

        suffixes = [
            "_family_summary",
            "_10d_to_1d",
            "_20d_to_1d",
            "_50d_to_1d",
            "_5d_to_1d",
            "_3d_to_1d",
            "_1d_to_1d",
        ]

        for suffix in suffixes:
            if fam.endswith(suffix):
                fam = fam[: -len(suffix)]

        return fam

    def _load_experiment_details(self, symbol: str, family: str) -> List[Dict[str, Any]]:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return []

        pattern = f"{symbol}_{family}_*.json"
        files = sorted(reports_dir.glob(pattern))

        results: List[Dict[str, Any]] = []

        for path in files:
            if path.name.endswith("_family_summary.json"):
                continue

            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                compact = self._compact_experiment_json(path.name, data)
                results.append(compact)

            except Exception:
                continue

        return results[:10]

    def _compact_experiment_json(self, filename: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the fields the LLM needs.
        Handles flexible JSON structures.
        """
        stats = data.get("statistics", data.get("stats", data))
        decision = data.get("decision", {})

        return {
            "file": filename,
            "experiment_name": data.get("experiment_name") or data.get("name") or filename,
            "x_col": data.get("x_col"),
            "y_col": data.get("y_col"),
            "coefficient": self._find_first(stats, ["coefficient", "coef", "slope"]),
            "p_value": self._find_first(stats, ["p_value", "pvalue", "p-value"]),
            "r_squared": self._find_first(stats, ["r_squared", "r2", "R-squared"]),
            "observations": self._find_first(stats, ["observations", "nobs", "n"]),
            "decision": decision.get("action") if isinstance(decision, dict) else decision,
        }

    def _find_first(self, data: Any, keys: List[str]) -> Any:
        if not isinstance(data, dict):
            return None

        for key in keys:
            if key in data:
                return data[key]

        # Search one level down
        for value in data.values():
            if isinstance(value, dict):
                found = self._find_first(value, keys)
                if found is not None:
                    return found

        return None

    def _build_prompt(
        self,
        question: str,
        deterministic_answer: str,
        evidence: Dict[str, Any],
    ) -> str:
        history = self.conversation_history[-6:]

        return f"""
User question:
{question}

Recent conversation:
{json.dumps(history, indent=2)}

Active context (current comparison):
- Symbols: {self.active_context['symbols']}
- Family: {self.active_context['family']}

Deterministic answer/context:
{deterministic_answer}

Grounded experiment evidence:
{json.dumps(evidence, indent=2, default=str)}

Answer instructions:
1. Answer the user's exact question.
2. Use the deterministic decision/classification as the authority.
3. Explain the result using the provided metrics only.
4. Mention missing data when relevant.
5. Do not introduce untested concepts.
6. Keep it concise.
7. For follow-up questions without explicit symbols, use active context to answer based on current comparison.
8. For low R²: explain that signal explains only tiny fraction of future returns, so practical predictive strength is weak even if p-value significant.
"""

    def _get_rule_based_explanation(self, question: str, context: str) -> str:
        q = question.lower()
        evidence = self._collect_grounded_evidence(question)

        lines = ["🤖 Rule-Based Grounded Explanation:", ""]

        if not evidence.get("matched_symbols"):
            lines.append("I need a symbol from the available reports to ground the answer.")
            lines.append("")
            lines.append("Example:")
            lines.append("  why is MSFT REFINE for ma_distance_reversion?")
            return "\n".join(lines)

        for symbol, families in evidence.get("symbols", {}).items():
            for family, data in families.items():
                summary = data.get("family_summary", {})
                stability = data.get("stability", {})
                experiments = data.get("experiments", [])

                decision = summary.get("decision", {})
                action = decision.get("action", "UNKNOWN") if isinstance(decision, dict) else decision

                lines.append(f"📊 {symbol} / {family}")
                lines.append(f"Decision: {action}")
                lines.append("")

                if experiments:
                    lines.append("Evidence:")
                    for exp in experiments[:5]:
                        lines.append(
                            f"- {exp.get('experiment_name')}: "
                            f"coef={exp.get('coefficient')}, "
                            f"p={exp.get('p_value')}, "
                            f"R²={exp.get('r_squared')}"
                        )
                    lines.append("")

                survival = stability.get("significance_survival_rate")
                direction = stability.get("direction_consistent")

                if survival is not None:
                    lines.append(f"Stability survival rate: {survival}%")
                if direction is not None:
                    lines.append(f"Direction consistent: {direction}")

                lines.append("")

                if action == "REFINE":
                    lines.append("Interpretation:")
                    lines.append(
                        "The signal has some evidence, but the deterministic system does not trust it enough to promote."
                    )
                    if survival == 0.0:
                        lines.append(
                            "The key issue is that significance does not survive the train/test split."
                        )
                    lines.append("So REFINE means: promising, but not robust enough yet.")

                elif action == "DROP":
                    lines.append("Interpretation:")
                    lines.append(
                        "The signal failed the minimum reliability checks. It may have weak or borderline evidence, but not enough to rely on."
                    )

                elif action == "PROMOTE":
                    lines.append("Interpretation:")
                    lines.append(
                        "The signal passed the system's robustness checks and is considered strong enough to promote."
                    )

                return "\n".join(lines)

        return context


def main() -> None:
    print("🤖 LLM-Enhanced Experiment Results Chat")
    print("=" * 60)
    print("Deterministic results + grounded AI explanations.")
    print("Type 'quit' to exit, 'help' for available data.")
    print()

    chat = LLMChatInterface()

    while True:
        try:
            question = input("❓ Your question: ").strip()

            if question.lower() in {"quit", "exit", "q"}:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            print()
            print(chat.process_question_with_llm(question))
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()