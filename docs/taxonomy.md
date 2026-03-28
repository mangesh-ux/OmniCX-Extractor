# OmniCX Taxonomy and Labeling Contract

This document is the canonical taxonomy and field contract for OmniCX extraction outputs.

Primary implementation source: `src/schema.py` (`LogisticsCXMetrics`).

## Top-Level Output Groups

Every prediction must return:
- `behavioral_analytics`
- `operational_analytics`
- `diagnostic_reasoning`

## 1) Behavioral Analytics

Captures customer effort, friction, intent, and conversational dynamics.

### Fields

- `effort_and_friction_quotes` (`List[str]`)
  - Extract 1-3 direct quotes evidencing frustration/rework/effort.
  - Use `[]` when no explicit friction evidence exists.

- `customer_intent` (`CustomerIntent`)
  - Allowed intents:
    - `WISMO_Standard`
    - `WISMO_Stalled_Label`
    - `Intercept_Request`
    - `Address_Modification`
    - `Delivery_Scheduling`
    - `Proof_of_Delivery_Dispute`
    - `Damage_Claim_Initiation`
    - `Lost_in_Transit_Investigation`
    - `Hazmat_Documentation_Hold`
    - `Customs_Clearance_Inquiry`

- `customer_effort_score` (`int`, 1-5)
  - Rubric:
    - `1`: Zero effort; immediate resolution; no repetition
    - `2`: Low effort; routine transaction; minor friction
    - `3`: Moderate effort; one clarification/repeat
    - `4`: High effort; explicit frustration or handling struggle
    - `5`: Extreme effort; escalation/channel switching/failure to resolve

- `sentiment_trajectory` (`SentimentTrajectory`)
  - Allowed values:
    - `Improved - Customer ended calmer or more satisfied than they started`
    - `Worsened - Customer ended more frustrated or angry than they started`
    - `Unchanged - Customer maintained the exact same emotional state throughout`

- `rework_frequency` (`ReworkFrequency`)
  - Allowed values:
    - `0 - No rework, customer stated issue once`
    - `1 - Customer had to repeat or clarify themselves exactly once`
    - `2+ - Chronic rework, customer had to repeatedly correct the agent or restate facts`

## 2) Operational Analytics

Captures logistics exception diagnosis, root-cause family, boolean control flags, and resolution state.

### Fields

- `exception_diagnostic_reasoning` (`str`)
  - Evidence-grounded explanation from transcript.
  - Use `"None"` when no explicit exception reasoning is present.

- `delivery_exception_type` (`DeliveryExceptionType`)
  - Current implemented enum values:
    - `DEX08 - Recipient Not In/Business Closed`
    - `STAT84 - Delay Beyond our Control`
    - `Unknown / Not Explicitly Stated`
  - Policy: if transcript lacks explicit code/reason, use `Unknown / Not Explicitly Stated`.

- `root_cause_category` (`RootCauseCategory`)
  - Allowed values:
    - `Address, Location, and Recipient Failures`
    - `Environmental and Force Majeure Events`
    - `Operational, Mechanical, and Technological Failures`
    - `Documentation and Labeling Deficiencies`
    - `Hazardous Materials (Hazmat) and Dangerous Goods Violations`
    - `Unknown / Not Applicable`

- Deterministic booleans:
  - `address_change_requested`
  - `missed_delivery_explicitly_mentioned`
  - `escalation_requested`
  - `agent_explicitly_confirmed_resolution`

- `unresolved_next_steps` (`str`)
  - If unresolved: concrete pending action (`"Ticket submitted to hub"`, `"Waiting on customs"`, etc.)
  - If resolved: `"N/A"`

### Resolution semantics

- `agent_explicitly_confirmed_resolution = true` only if the agent explicitly confirms fix/resolution in current interaction.
- `unresolved_next_steps = "N/A"` when resolved; otherwise must name concrete next action.

## 3) Diagnostic Reasoning

Auditability and explanation fields.

### Fields

- `recommended_routing_queue` (`str`)
- `intent_reasoning` (`str`)
- `exception_reasoning` (`str`)
- `effort_reasoning` (`str`)

These fields should explain why classification choices were made, grounded in transcript evidence.

## Tie-Break and Ambiguity Policy

- Prioritize transcript evidence over assumptions.
- If exception details are ambiguous, default to `Unknown / Not Explicitly Stated`.
- If root-cause family is unclear, default to `Unknown / Not Applicable`.
- Do not infer resolution without explicit confirmation.

## Knowledge Provenance

Taxonomy design is informed by:
- `docs/knowledge/Transcript-Only CX Difficulty Score_ Standards, Methods, and a Rigorous MVP Design.pdf`
- `docs/knowledge/Logistics CX Data Schema Development.docx`

Canonical code-level enforcement remains in `src/schema.py`.

## Synthetic Data Generation Model Provenance

For the synthetic data pipeline in this repository:
- Transcript generation model: `gpt-4o-mini` (implemented in `src/data_factory.py`)
- Schema-constrained labeling model: `gpt-4o-mini` (implemented in `src/extractor.py`)

If generation/labeling models are changed in future iterations, update this section and corresponding cards for reproducibility.
