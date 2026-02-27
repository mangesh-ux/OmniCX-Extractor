from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. ANALYTICS ENUMERATIONS (The "Group By" Variables)
# ==========================================

class DeliveryExceptionType(str, Enum):
    # [Note: Insert the 90 STAT/DEX/PUX/REX codes here from the previous step]
    # Example truncated for brevity:
    DEX08 = "DEX08 - Recipient Not In/Business Closed"
    STAT84 = "STAT84 - Delay Beyond our Control"
    UNKNOWN = "Unknown / Not Explicitly Stated"

class CustomerIntent(str, Enum):
    WISMO_STANDARD = "WISMO_Standard"
    WISMO_STALLED_LABEL = "WISMO_Stalled_Label"
    INTERCEPT_REQUEST = "Intercept_Request"
    ADDRESS_MODIFICATION = "Address_Modification"
    DELIVERY_SCHEDULING = "Delivery_Scheduling"
    PROOF_OF_DELIVERY_DISPUTE = "Proof_of_Delivery_Dispute"
    DAMAGE_CLAIM_INITIATION = "Damage_Claim_Initiation"
    LOST_IN_TRANSIT_INVESTIGATION = "Lost_in_Transit_Investigation"
    HAZMAT_DOCUMENTATION_HOLD = "Hazmat_Documentation_Hold"
    CUSTOMS_CLEARANCE_INQUIRY = "Customs_Clearance_Inquiry"

class Sentiment(str, Enum):
    CALM_NEUTRAL = "Calm/Neutral"
    MILD_FRICTION = "Mild Friction"
    STRONG_NEGATIVE = "Strong Negative/Frustrated"

class RootCauseCategory(str, Enum):
    ADDRESS_LOCATION_RECIPIENT_FAILURES = "Address, Location, and Recipient Failures"
    ENVIRONMENTAL_FORCE_MAJEURE = "Environmental and Force Majeure Events"
    OPERATIONAL_MECHANICAL_TECHNOLOGICAL = "Operational, Mechanical, and Technological Failures"
    DOCUMENTATION_LABELING_DEFICIENCIES = "Documentation and Labeling Deficiencies"
    HAZARDOUS_MATERIALS_VIOLATIONS = "Hazardous Materials (Hazmat) and Dangerous Goods Violations"
    UNKNOWN_OR_NOT_APPLICABLE = "Unknown / Not Applicable"

# ==========================================
# 2. DOWNSTREAM ANALYTICS MODELS (Quantitative)
# ==========================================

class SentimentTrajectory(str, Enum):
    IMPROVED = "Improved - Customer ended calmer or more satisfied than they started"
    WORSENED = "Worsened - Customer ended more frustrated or angry than they started"
    UNCHANGED_FLAT = "Unchanged - Customer maintained the exact same emotional state throughout"

class ReworkFrequency(str, Enum):
    NONE = "0 - No rework, customer stated issue once"
    SINGLE = "1 - Customer had to repeat or clarify themselves exactly once"
    MULTIPLE = "2+ - Chronic rework, customer had to repeatedly correct the agent or restate facts"

class BehavioralAnalytics(BaseModel):
    """Metrics quantifying the psychological toll and effort, forced through Chain-of-Thought."""
    
    # 1. Evidence extracted FIRST
    effort_and_friction_quotes: List[str] = Field(..., description="Extract 1-3 direct quotes where the customer expresses frustration, repeats themselves, or complains about the process. If none, return empty list.")
    
    # 2. Rubric-based Classifications SECOND
    customer_intent: CustomerIntent = Field(..., description="The primary reason the customer initiated contact.")
    
    # The CES Rubric - Zero ambiguity allowed
    customer_effort_score: int = Field(..., ge=1, le=5, description="""
        Assign strict effort score based ONLY on this rubric:
        1 = Zero effort: Issue resolved instantly, no repetition required.
        2 = Low effort: Routine transaction, minor wait time, no frustration.
        3 = Moderate effort: Customer had to clarify or repeat themselves once, mild friction.
        4 = High effort: Customer showed explicit frustration, agent struggled to understand or resolve.
        5 = Extreme effort: Escalation, channel switching mentioned, or complete failure to resolve causing anger.
    """)
    
    sentiment_trajectory: SentimentTrajectory = Field(..., description="The directional change in the customer's mood from the first message to the last.")
    
    rework_frequency: ReworkFrequency = Field(..., description="Categorize how many times the customer had to repeat themselves or correct the agent.")
    
class OperationalAnalytics(BaseModel):
    """Metrics diagnosing the physical supply chain failure, validated by textual evidence."""
    
    # 1. Evidence extracted FIRST
    exception_diagnostic_reasoning: str = Field(..., description="Quote the agent explaining WHY the package is delayed or what the exception is. If not stated, write 'None'.")
    
    # 2. Classifications generated SECOND
    delivery_exception_type: DeliveryExceptionType = Field(default=DeliveryExceptionType.UNKNOWN, description="MUST align with the diagnostic reasoning above. If the agent didn't specify a code/exact reason, pick UNKNOWN.")
    root_cause_category: RootCauseCategory = Field(default=RootCauseCategory.UNKNOWN_OR_NOT_APPLICABLE, description="High-level category of the operational failure.")
    
    # 3. Deterministic Flags
    address_change_requested: bool = Field(False, description="True ONLY if the customer explicitly asked to reroute or change the address.")
    missed_delivery_explicitly_mentioned: bool = Field(False, description="True ONLY if a failed delivery attempt is discussed.")
    escalation_requested: bool = Field(False, description="True ONLY if the customer explicitly asked for a manager, supervisor, or tier-2 transfer.")
    
    # 4. Rigorous Resolution Tracking
    agent_explicitly_confirmed_resolution: bool = Field(..., description="True ONLY if the agent explicitly states the issue is fixed/resolved on this interaction.")
    unresolved_next_steps: str = Field(..., description="If not resolved, what is the pending action? (e.g., 'Ticket submitted to hub', 'Waiting on customs'). If resolved, write 'N/A'.")

# ==========================================
# 3. VALIDATION & REASONING MODEL (Qualitative)
# ==========================================

class DiagnosticReasoning(BaseModel):
    """Textual outputs allowing Decision Scientists to audit the LLM's logic."""
    recommended_routing_queue: str = Field(..., description="Suggested support queue for this intent (e.g., 'Tier 2 Retention', 'Hazmat Desk').")
    intent_reasoning: str = Field(..., description="Quote the specific text from the transcript that justifies the selected customer_intent.")
    exception_reasoning: str = Field(..., description="Explain why the specific delivery_exception_type and root_cause_category were chosen.")
    effort_reasoning: str = Field(..., description="Briefly explain what conversational cues drove the customer_effort_score rating.")

# ==========================================
# MAIN SCHEMA MODEL
# ==========================================

class LogisticsCXMetrics(BaseModel):
    """
    Lean, compute-efficient extraction schema containing only cognitive analytics and reasoning.
    (Metadata like tracking numbers and timestamps are joined from the CRM database post-extraction).
    """
    behavioral_analytics: BehavioralAnalytics
    operational_analytics: OperationalAnalytics
    diagnostic_reasoning: DiagnosticReasoning