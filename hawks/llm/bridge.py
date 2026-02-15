"""LLM Trust Bridge — condition LLM responses on operator trust state."""

from __future__ import annotations

import json
import os

import anthropic


class LLMTrustBridge:
    """Bridge HAWKS trust dynamics with an LLM via the Anthropic SDK."""

    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model_name

    def generate_response(self, trust: float, ai_confidence: float, structural_risk: float) -> dict:
        """Generate a trust-conditioned LLM response.

        Parameters
        ----------
        trust : float
            Operator trust level in [0, 1].
        ai_confidence : float
            AI model confidence in [0, 1].
        structural_risk : float
            Structural risk factor in [0, 1].

        Returns
        -------
        dict
            Keys: ``belief_statement``, ``justification``, ``decision``.
        """
        if trust < 0.4:
            tone = "skeptical, critical"
        elif trust >= 0.7:
            tone = "accepting, supportive"
        else:
            tone = "cautious, balanced"

        system_prompt = (
            f"You are a manufacturing safety analyst. "
            f"The current operator trust level is {trust:.2f}. "
            f"Respond in a {tone} tone that reflects this trust level."
        )

        user_message = (
            f"Given the following parameters:\n"
            f"- Trust level (T): {trust}\n"
            f"- AI confidence (C): {ai_confidence}\n"
            f"- Structural risk (R): {structural_risk}\n\n"
            f"Respond with STRICT JSON only, no other text:\n"
            f'{{"belief_statement": "<your belief>", '
            f'"justification": "<your justification>", '
            f'"decision": "accept" or "reject"}}'
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "belief_statement": raw_text,
                "justification": "LLM response was not valid JSON",
                "decision": "reject",
            }


if __name__ == "__main__":
    api_key = os.environ["ANTHROPIC_API_KEY"]
    bridge = LLMTrustBridge(api_key=api_key)
    result = bridge.generate_response(trust=0.2, ai_confidence=0.85, structural_risk=0.7)
    print(json.dumps(result, indent=2))
