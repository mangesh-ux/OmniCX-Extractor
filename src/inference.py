"""
Inference with the finetuned Qwen2.5-3B LoRA for logistics CX extraction.

Load the model once, then call extract_with_finetuned(transcript) to get
LogisticsCXMetrics (or a dict). Use in scripts, notebooks, or the local server.

Locked: 2025-02 — API stable; system_prompt overridable for server/experiments.
"""
import unsloth  # noqa: F401  (must be before trl/transformers)

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Union

# Avoid Transformers 5.x + Unsloth logging TypeError: the deprecation warning for
# AttentionMaskConverter is passed with an extra arg and breaks the logger's format.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.modeling_attn_mask_utils",
)
try:
    import transformers.utils.logging as tx_logging
    tx_logging.set_verbosity_error()
except Exception:
    pass

# Repo root and src on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from schema import LogisticsCXMetrics

# Same system prompt as training (must match exactly)
SYSTEM_PROMPT = (
    "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."
)

# Default LoRA path (relative to repo root)
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "qwen-logistics-lora"


def _strip_json_block(raw: str) -> str:
    """Remove optional ```json ... ``` wrapper."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()
    return raw


def load_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """
    Load the finetuned model and tokenizer (Unsloth).
    Call once, then pass model/tokenizer to extract_with_finetuned.

    If the path contains only LoRA adapter files (adapter_config.json), loads
    the base model first then applies the adapter. Otherwise loads from path directly.

    Returns:
        (model, tokenizer)
    """
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        # Adapter-only save: load base then apply LoRA with PEFT
        from peft import PeftModel
        from transformers import AutoTokenizer

        model, _ = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        model = PeftModel.from_pretrained(model, str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


def extract_with_finetuned(
    transcript: str,
    model=None,
    tokenizer=None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    system_prompt: Union[str, None] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    return_dict: bool = False,
) -> Union[LogisticsCXMetrics, dict]:
    """
    Run extraction on a single transcript using the finetuned LoRA.

    Args:
        transcript: Raw conversation text (e.g. "Agent: ... Customer: ...").
        model, tokenizer: From load_model(). If None, loads from model_path (slower per call).
        model_path: Where the LoRA is saved (used only if model/tokenizer not provided).
        system_prompt: Override the default system prompt (for experiments). None = use SYSTEM_PROMPT.
        max_new_tokens: Max length of generated JSON.
        temperature: Lower = more deterministic.
        return_dict: If True, return a dict; otherwise return LogisticsCXMetrics.

    Returns:
        LogisticsCXMetrics or dict (if return_dict=True or validation fails).
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    prompt_text = (system_prompt if system_prompt is not None else SYSTEM_PROMPT).strip()
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": f"Transcript:\n{transcript.strip()}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Decode only the new tokens (assistant reply)
    reply = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    json_str = _strip_json_block(reply)
    try:
        data = json.loads(json_str)
        metrics = LogisticsCXMetrics.model_validate(data)
        return metrics if not return_dict else metrics.model_dump()
    except Exception as e:
        # Return raw dict for manual inspection if validation fails
        try:
            return json.loads(json_str) if not return_dict else json.loads(json_str)
        except Exception:
            raise ValueError(f"Model output could not be parsed as JSON. Raw reply:\n{reply[:500]}...") from e


def run_example():
    """Load model, run on a short example, print result."""
    example = """
**Subject: Proof of Delivery Dispute**\n\n---\n\n**Customer:**  \nHello,  \nI need help. I order package, but it not come to my house. The proof say delivered, but I no see it. My address is [Customer's Address]. Can you find my package, please?  \n\n---\n\n**Agent:**  \nDear [Customer's Name],  \nThank you for reaching out to us regarding your package delivery. I understand how important it is to receive your items on time. Could you please confirm if you have checked with your neighbors or around your property in case the package was left somewhere else?  \n\n---\n\n**Customer:**  \nI check outside. No package. I ask neighbor. They say no see. I look everywhere. When I track, it say delivered but to wrong place. This make me very worry.  \n\n---\n\n**Agent:**  \nI appreciate your patience, [Customer's Name]. Based on the tracking information, it appears that the package may have been delivered to an address other than yours. This issue is known as a misdelivery (DEX04). I will investigate this further. Please allow me some time, and I will update you shortly.  \n\n---\n\n**Customer:**  \nOkay, thank you. I wait. I need my item. It very important for me.  \n\n---\n\n**Agent:**  \nI completely understand, and I assure you that I am doing everything I can to resolve this issue quickly. I will contact the delivery team and see if they can provide more details about where the package was delivered. Please expect my next update within 24 hours.  \n\n---\n\n**Customer:**  \nThank you. I hope you find it soon. I no know what to do without my package.  \n\n---\n\n**Agent:**  \nYou're welcome, [Customer's Name]. I truly empathize with your situation, and I will keep you updated as soon as I have more information. Thank you for your understanding and patience.  \n\n---\n\n**Customer:**  \nOkay, I wait. Thank you.  \n\n---\n\n**Agent:**  \nThank you for your response, [Customer's Name]. I will be in touch soon with an update. Have a great day!  \n\n---"}, {"role": "assistant", "content": "{\"behavioral_analytics\":{\"effort_and_friction_quotes\":[\"I check outside. No package.\",\"When I track, it say delivered but to wrong place. This make me very worry.\",\"I no know what to do without my package.\"],\"customer_intent\":\"Proof_of_Delivery_Dispute\",\"customer_effort_score\":4,\"sentiment_trajectory\":\"Worsened - Customer ended more frustrated or angry than they started\",\"rework_frequency\":\"0 - No rework, customer stated issue once\"},\"operational_analytics\":{\"exception_diagnostic_reasoning\":\"Based on the tracking information, it appears that the package may have been delivered to an address other than yours.\",\"delivery_exception_type\":\"Unknown / Not Explicitly Stated\",\"root_cause_category\":\"Address, Location, and Recipient Failures\",\"address_change_requested\":false,\"missed_delivery_explicitly_mentioned\":false,\"escalation_requested\":false,\"agent_explicitly_confirmed_resolution\":false,\"unresolved_next_steps\":\"Investigating misdelivery with the delivery team\"},\"diagnostic_reasoning\":{\"recommended_routing_queue\":\"Tier 2 Delivery Issues\",\"intent_reasoning\":\"I need help. I order package, but it not come to my house.\",\"exception_reasoning\":\"The package may have been delivered to an address other than yours, indicating a misdelivery.\",\"effort_reasoning\":\"The customer expressed significant worry and frustration about the package not being delivered correctly.
"""
    print("Loading model...")
    model, tokenizer = load_model()
    print("Running extraction...")
    result = extract_with_finetuned(example, model=model, tokenizer=tokenizer)
    print(result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else json.dumps(result, indent=2))


if __name__ == "__main__":
    run_example()
