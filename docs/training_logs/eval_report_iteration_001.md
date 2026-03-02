# Evaluation Report — Iteration 001

**Eval file:** `eval_dataset.jsonl`  
**Total examples:** 32  
**Valid (no runtime error):** 32  
**Errors:** 0  

## Summary


| Metric                                 | Value       |
| -------------------------------------- | ----------- |
| Strict accuracy (all key fields match) | 0.0% (0/32) |


## Per-field accuracy


| Field                                 | Correct | Accuracy |
| ------------------------------------- | ------- | -------- |
| customer_intent                       | 17/32   | 53.1%    |
| customer_effort_score                 | 12/32   | 37.5%    |
| sentiment_trajectory                  | 21/32   | 65.6%    |
| rework_frequency                      | 19/32   | 59.4%    |
| delivery_exception_type               | 20/32   | 62.5%    |
| root_cause_category                   | 17/32   | 53.1%    |
| address_change_requested              | 32/32   | 100.0%   |
| missed_delivery_explicitly_mentioned  | 28/32   | 87.5%    |
| escalation_requested                  | 32/32   | 100.0%   |
| agent_explicitly_confirmed_resolution | 25/32   | 78.1%    |
| unresolved_next_steps                 | 26/32   | 81.2%    |


## Failures (strict)

- **Line 1**: missed customer_effort_score, sentiment_trajectory, delivery_exception_type, root_cause_category
- **Line 2**: missed customer_intent, customer_effort_score, delivery_exception_type, missed_delivery_explicitly_mentioned
- **Line 3**: missed delivery_exception_type
- **Line 4**: missed customer_intent, rework_frequency
- **Line 5**: missed sentiment_trajectory, rework_frequency
- **Line 6**: missed customer_intent, customer_effort_score, root_cause_category
- **Line 7**: missed customer_effort_score, rework_frequency, missed_delivery_explicitly_mentioned
- **Line 8**: missed customer_intent, root_cause_category, agent_explicitly_confirmed_resolution, unresolved_next_steps
- **Line 9**: missed customer_intent, customer_effort_score, delivery_exception_type, root_cause_category
- **Line 10**: missed customer_intent, customer_effort_score, delivery_exception_type, missed_delivery_explicitly_mentioned
- **Line 11**: missed agent_explicitly_confirmed_resolution, unresolved_next_steps
- **Line 12**: missed customer_intent, sentiment_trajectory, rework_frequency, root_cause_category, missed_delivery_explicitly_mentioned
- **Line 13**: missed rework_frequency, agent_explicitly_confirmed_resolution
- **Line 14**: missed customer_intent, customer_effort_score, delivery_exception_type
- **Line 15**: missed customer_intent, agent_explicitly_confirmed_resolution
- **Line 16**: missed customer_intent, customer_effort_score, root_cause_category, agent_explicitly_confirmed_resolution, unresolved_next_steps
- **Line 17**: missed customer_intent, customer_effort_score, sentiment_trajectory, rework_frequency, agent_explicitly_confirmed_resolution, unresolved_next_steps
- **Line 18**: missed customer_effort_score, rework_frequency, root_cause_category, agent_explicitly_confirmed_resolution, unresolved_next_steps
- **Line 19**: missed customer_intent, sentiment_trajectory, rework_frequency, root_cause_category
- **Line 20**: missed customer_intent, customer_effort_score, delivery_exception_type, root_cause_category
- **Line 21**: missed customer_effort_score, sentiment_trajectory
- **Line 22**: missed customer_effort_score, delivery_exception_type, root_cause_category
- **Line 23**: missed customer_effort_score, sentiment_trajectory, rework_frequency, delivery_exception_type, root_cause_category
- **Line 24**: missed customer_effort_score, sentiment_trajectory, delivery_exception_type
- **Line 25**: missed customer_effort_score, sentiment_trajectory, rework_frequency, root_cause_category
- **Line 26**: missed sentiment_trajectory, rework_frequency, root_cause_category
- **Line 27**: missed customer_effort_score, root_cause_category
- **Line 28**: missed customer_effort_score, sentiment_trajectory
- **Line 29**: missed customer_intent, rework_frequency, delivery_exception_type
- **Line 30**: missed customer_intent, root_cause_category
- **Line 31**: missed customer_effort_score, rework_frequency, delivery_exception_type, unresolved_next_steps
- **Line 32**: missed customer_effort_score

