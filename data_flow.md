# HAWKS Data Flow: Three-Layer Architecture

## Layer 1: Mechanistic (No Neural Network)

This is the core simulation loop (`hawks/engine/simulation.py`). Pure closed-form equations, no learning.

```
PhysicsEngine          →  structural_risk (float)
     ↓
AIModelSimulator       →  confidence, flagged, correct (floats + bools)
     ↓
TrustModel.update()    →  T_{t+1} (float)
     ↓
DecisionModel.decide() →  "accept" / "reject"
```

### Concrete Handoff at Each Step

1. `PhysicsEngine.simulate_print()` outputs a `PrintTrace` with `max_structural_risk`
2. That `structural_risk` feeds into `AIModelSimulator.inspect_part()` which produces `Detection(confidence, flagged)`
3. Both `trust` + `detection.confidence` + `structural_risk` feed into `DecisionModel.decide()` which returns accept/reject
4. Ground truth feeds back into `TrustModel.update(ai_was_correct)` which updates T for the next step

All of this is closed-form math. Sigmoids with hand-set weights, not learned weights.

### Key Source Files

- `hawks/manufacturing/physics_engine.py` — 9-state physics simulation
- `hawks/detection/ai_model.py` — AI classification with TPR/FPR and calibrated confidence
- `hawks/operators/trust.py` — Asymmetric Bayesian trust update (alpha/beta)
- `hawks/operators/decision.py` — Sigmoid-based binary accept/reject

---

## Layer 2: Neural Surrogate (Small NN)

The surrogate (`hawks/models/surrogate.py`) learns to replicate Layer 1's trust dynamics.

```
Layer 1 simulation runs → SyntheticDataGenerator.generate() → DataFrame
     ↓
DataFrame columns: [structural_risk, ai_confidence, ai_correctness, operator_trust]
     ↓
SyntheticDataGenerator.to_tensors() → (N, 4) float32 tensor
     ↓
TrustSurrogate MLP (4→32→16→1) trains on this → predicts T_{t+1}
```

Layer 1 generates the training data. Layer 2 learns to approximate it. The surrogate does not feed back into Layer 1 — it is a parallel path that proves the dynamics are learnable.

### Key Source Files

- `hawks/data/synthetic.py` — Generates ML-ready DataFrames from simulation
- `hawks/models/surrogate.py` — Two-hidden-layer MLP (4→32→16→1, ReLU, Sigmoid output)

---

## Layer 3: LLM Bridge (Generative)

The LLM bridge (`hawks/llm/bridge.py`) takes Layer 1's state variables and conditions the LLM.

```
Layer 1 produces:  trust=0.2, ai_confidence=0.85, structural_risk=0.7
     ↓
LLMTrustBridge maps trust → tone:
   T < 0.4  → "skeptical, critical"
   T >= 0.7 → "accepting, supportive"
   else     → "cautious, balanced"
     ↓
System prompt: "operator trust level is 0.20, respond in skeptical tone"
User prompt:   "T=0.2, C=0.85, R=0.7 → give belief, justification, decision"
     ↓
Claude API call → JSON: {belief_statement, justification, decision}
```

### Key Source Files

- `hawks/llm/bridge.py` — Anthropic SDK integration, trust-conditioned prompting

---

## How the Three Layers Come Together

```
                    Layer 1 (Mechanistic)
                    ┌─────────────────────┐
                    │ Physics → AI → Trust │
                    │   → Decision Loop    │
                    └──────┬──────┬───────┘
                           │      │
              training data│      │ live state (T, C, R)
                           │      │
                    ┌──────▼──┐ ┌─▼──────────┐
                    │ Layer 2 │ │  Layer 3    │
                    │ Surrogate│ │ LLM Bridge │
                    │ (MLP)   │ │ (Claude)   │
                    └─────────┘ └────────────┘
                    "can we     "what would
                     learn       this operator
                     this?"      actually say?"
```

The three layers are **not a sequential pipeline**. Layer 1 is the source of truth. Layer 2 and Layer 3 both consume Layer 1's outputs independently:

- **Layer 2** asks: "Can a small NN learn these dynamics from data alone?"
- **Layer 3** asks: "Given this trust state, what would a human operator actually say?"

Neither Layer 2 nor Layer 3 feeds back into Layer 1. This is a deliberate design choice — it keeps the mechanistic model interpretable and uncontaminated by learned or generated behavior.
