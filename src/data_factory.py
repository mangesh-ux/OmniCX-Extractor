"""
Autonomous Synthetic Data Generation pipeline for logistics CX.
Generates realistic multi-turn transcripts, then runs them through extract_logistics_data
and appends JSONL rows to data/processed/golden_training_dataset.jsonl for fine-tuning.
"""

import json
import random
import sys
import time
from pathlib import Path

# Ensure src is on path for imports when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from openai import OpenAI
import dotenv
from tqdm import tqdm

from extractor import extract_logistics_data

# ---------------------------------------------------------------------------
# RANDOMIZED VARIABLES (Impedance Mismatch Matrix + Channels + Personas)
# ---------------------------------------------------------------------------

channels = [
    "Voice Call (include spoken interruptions, IVR mentions, 'can you hear me', holds)",
    "Live Web Chat (include typos, short fragmented messages, informal tone)",
    "Email Thread (asynchronous, formal replies, time gaps mentioned)",
]

customer_personas = [
    "Furious & High Urgency (e.g., missed medical supplies or passport; highly emotional)",
    "Confused & Tech-Illiterate (e.g., elderly; doesn't understand the app or tracking portal)",
    "Calm but Analytical (e.g., has Ring camera footage proving driver didn't stop; demands proof)",
    "The Third-Party Recipient (e.g., receiving a gift; doesn't have the sender's billing info)",
    "Small Business / B2B (e.g., shipping in bulk; angry about SLAs; uses terms like 'freight')",
    "Terse & Impatient (e.g., one-word answers; refuses to verify address easily; wants it fixed NOW)",
    "Verbose & Chatty (e.g., over-explains the backstory, buries the actual problem in long paragraphs)",
    "Language Barrier (e.g., uses simple vocabulary, broken grammar; struggles to explain physical locations)",
]

# [Customer Intent] + [Underlying Operational Reality]
logistics_scenarios = [
    "Intent: WISMO (Where is my order) | Reality: DEX08 (Recipient Not In/Business Closed)",
    "Intent: WISMO (Where is my order) | Reality: STAT50 (Customs Hold/Missing Commercial Invoice)",
    "Intent: Address Modification | Reality: PUX24 (Customer Delay - Package already on truck, cannot intercept)",
    "Intent: Proof of Delivery Dispute | Reality: DEX04 (Delivered to Address Other than Recipient / Misdelivered)",
    "Intent: Damage Claim Initiation | Reality: STAT37 (Observed Package Damage at Hub prior to delivery)",
    "Intent: Delivery Scheduling | Reality: HAL (Hold at Location successfully processed)",
    "Intent: Lost in Transit Investigation | Reality: MIS (Missort routing loop between two hubs)",
    "Intent: WISMO (Where is my order) | Reality: STAT84 (Weather/Force Majeure Delay)",
]

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REPO_ROOT = _SCRIPT_DIR.parent
OUTPUT_JSONL = REPO_ROOT / "data" / "processed" / "golden_training_dataset.jsonl"
SYSTEM_PROMPT = "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2
NUM_SAMPLES_DEFAULT = 10


def get_client() -> OpenAI:
    dotenv.load_dotenv(REPO_ROOT / ".env")
    return OpenAI(api_key=__import__("os").environ.get("OPENAI_API_KEY"))


def generate_transcript(client: OpenAI, channel: str, persona: str, scenario: str) -> str:
    """Ask gpt-4o-mini for a single multi-turn transcript matching the given combo."""
    user_prompt = f"""Generate a single, realistic logistics customer-support transcript with 6–10 turns.

Constraints:
- Channel: {channel}
- Customer persona: {persona}
- Scenario (intent vs operational reality): {scenario}

Rules:
- The CUSTOMER must speak only in layman's terms (no internal codes like DEX08, STAT50, etc.). They describe the situation in plain language.
- The AGENT diagnoses the issue professionally and with empathy, and may use internal codes in their own reasoning/notes only if appropriate for the channel (e.g., not read aloud on a call in a robotic way).
- Format each turn clearly as "Agent:" or "Customer:" followed by the message.
- Make the conversation coherent and specific to the scenario and persona."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at writing realistic customer support dialogue for logistics and shipping companies. Output only the transcript, no preamble."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.85,
    )
    return (response.choices[0].message.content or "").strip()


def build_jsonl_row(transcript: str, extracted_json: str) -> dict:
    """Build one JSONL row in fine-tuning format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": extracted_json},
        ]
    }


def run_pipeline(num_samples: int = NUM_SAMPLES_DEFAULT) -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    backoff = INITIAL_BACKOFF_SEC

    for i in tqdm(range(num_samples), desc="Synthetic samples", unit="sample"):
        channel = random.choice(channels)
        persona = random.choice(customer_personas)
        scenario = random.choice(logistics_scenarios)

        transcript = None
        for attempt in range(MAX_RETRIES):
            try:
                transcript = generate_transcript(client, channel, persona, scenario)
                if not transcript:
                    raise ValueError("Empty transcript returned")
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    tqdm.write(f"Sample {i + 1}: Generation failed after {MAX_RETRIES} attempts: {e}")
                    transcript = None
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

        if transcript is None:
            backoff = INITIAL_BACKOFF_SEC
            continue

        extracted_json = None
        for attempt in range(MAX_RETRIES):
            try:
                metrics = extract_logistics_data(transcript)
                extracted_json = metrics.model_dump_json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    tqdm.write(f"Sample {i + 1}: Extraction failed after {MAX_RETRIES} attempts: {e}")
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

        if extracted_json is None:
            backoff = INITIAL_BACKOFF_SEC
            continue

        row = build_jsonl_row(transcript, extracted_json)
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        backoff = INITIAL_BACKOFF_SEC

    tqdm.write(f"Done. Appended {num_samples} rows to {OUTPUT_JSONL}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SAMPLES_DEFAULT
    run_pipeline(num_samples=n)
