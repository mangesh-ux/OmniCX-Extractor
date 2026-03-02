# ChatGPT Prompt: Generate 20-Point Evaluation Dataset (ChatML JSONL)

Copy the entire prompt below into ChatGPT. Save the model’s response as a single file: **one JSON object per line** (JSONL). No other text. Name the file `eval_dataset.jsonl` and place it in `data/eval/` in the project.

---

## Prompt (copy from here)

You are generating a **rigorous evaluation dataset** for a logistics CX extraction model. The model takes a customer-support transcript and outputs structured metrics in a fixed JSON schema.

**Your task:** Produce exactly **20 evaluation datapoints** in the format specified below. Each datapoint must be a single line of valid JSON (no newlines inside the line). The file format is JSONL: 20 lines, each line one complete JSON object.

**Required structure for each line (one JSON object):**

```json
{"messages": [{"role": "system", "content": "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."}, {"role": "user", "content": "<THE_TRANSCRIPT>"}, {"role": "assistant", "content": "<THE_EXTRACTED_JSON_AS_A_STRING>"}]}
```

- **system**: Use exactly this text (no changes):  
  `You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript.`
- **user**: The full transcript. Use realistic logistics/shipping support dialogue (Agent and Customer turns). Vary channels: voice-style, live chat (short messages, typos), and email-style (formal, asynchronous). Format clearly (e.g. "Agent: ..." and "Customer: ..." or "**Agent:**" and "**Customer:**").
- **assistant**: A single string that is valid JSON (escaped if needed). The JSON must match the schema below exactly. Use the **enum value strings** listed (e.g. for `customer_intent` use `"WISMO_Standard"` not a label).

**Schema for the assistant JSON (nested structure):**

- **behavioral_analytics**
  - `effort_and_friction_quotes`: array of strings (1–3 direct customer quotes showing frustration/repetition; empty array if none).
  - `customer_intent`: one of  
    `WISMO_Standard`, `WISMO_Stalled_Label`, `Intercept_Request`, `Address_Modification`, `Delivery_Scheduling`, `Proof_of_Delivery_Dispute`, `Damage_Claim_Initiation`, `Lost_in_Transit_Investigation`, `Hazmat_Documentation_Hold`, `Customs_Clearance_Inquiry`
  - `customer_effort_score`: integer 1–5 (1=resolved instantly, 5=extreme effort/escalation).
  - `sentiment_trajectory`: one of  
    `Improved - Customer ended calmer or more satisfied than they started`,  
    `Worsened - Customer ended more frustrated or angry than they started`,  
    `Unchanged - Customer maintained the exact same emotional state throughout`
  - `rework_frequency`: one of  
    `0 - No rework, customer stated issue once`,  
    `1 - Customer had to repeat or clarify themselves exactly once`,  
    `2+ - Chronic rework, customer had to repeatedly correct the agent or restate facts`

- **operational_analytics**
  - `exception_diagnostic_reasoning`: string (agent’s explanation of delay/exception, or "None").
  - `delivery_exception_type`: one of  
    `DEX08 - Recipient Not In/Business Closed`,  
    `STAT84 - Delay Beyond our Control`,  
    `Unknown / Not Explicitly Stated`
  - `root_cause_category`: one of  
    `Address, Location, and Recipient Failures`,  
    `Environmental and Force Majeure Events`,  
    `Operational, Mechanical, and Technological Failures`,  
    `Documentation and Labeling Deficiencies`,  
    `Hazardous Materials (Hazmat) and Dangerous Goods Violations`,  
    `Unknown / Not Applicable`
  - `address_change_requested`: boolean
  - `missed_delivery_explicitly_mentioned`: boolean
  - `escalation_requested`: boolean
  - `agent_explicitly_confirmed_resolution`: boolean
  - `unresolved_next_steps`: string (e.g. "Ticket submitted to hub" or "N/A" if resolved)

- **diagnostic_reasoning**
  - `recommended_routing_queue`: string (e.g. "Tier 2 Retention", "Hazmat Desk")
  - `intent_reasoning`: string (quote or phrase from transcript justifying customer_intent)
  - `exception_reasoning`: string (why delivery_exception_type and root_cause_category were chosen)
  - `effort_reasoning`: string (what drove the customer_effort_score)

**Scenario coverage (across the 20 datapoints):**

- Cover at least **6 different customer intents** (e.g. WISMO, address change, proof of delivery dispute, damage claim, lost in transit, customs).
- Include **varied effort**: at least 2 examples each of low (1–2), moderate (3), and high (4–5) customer effort.
- Include **at least 2** where the customer explicitly asks for a manager or escalation.
- Include **at least 2** where the agent explicitly confirms resolution and **at least 2** where the issue is left unresolved with clear next steps.
- Include **at least 1** proof-of-delivery / misdelivery scenario and **at least 1** customs or documentation hold.
- Vary **channels**: some voice-style (interruptions, “can you hear me”), some chat (short turns, typos), some email-style.
- Include **at least 1** language-barrier style (simple vocabulary, broken grammar) and **at least 1** very terse customer.

**Output rules:**

- Output **only** the 20 lines of JSON. No preamble, no explanation, no markdown code fence. Each line is one complete JSON object ending with `}`.
- In each object, the `"content"` of the assistant message must be a **single string** that is valid JSON when parsed (escape internal quotes as `\"`).
- Ensure every line is valid JSON and that the inner assistant JSON parses and contains all required keys above.

Generate the 20 datapoints now.

---

## After ChatGPT responds

1. Copy the model’s output (the 20 lines only).
2. Save as `data/eval/eval_dataset.jsonl` in the project (create `data/eval/` if needed).
3. Run evaluation:  
   `python scripts/run_evaluation.py --eval-file data/eval/eval_dataset.jsonl`
