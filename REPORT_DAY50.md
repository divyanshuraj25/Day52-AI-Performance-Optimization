# Day 50 - Product-Specific Evaluation: PayEasy Support Assistant

Run mode: **offline** | Timestamp: 20261029_105955

> Offline mode = deterministic extractive generator + rule-based judge (harness verification, no API calls). Re-run with `--mode openai` for real LLM numbers.

## 1. Tier breakdown (overall judge score, 0-1)

| Tier | Baseline | After fixes |
|---|---|---|
| easy | 0.82 | 0.94 |
| medium | 0.68 | 0.82 |
| hard | 0.47 | 0.76 |
| overall | 0.657 | 0.84 |

Weakest tier at baseline: **hard**.

## 2. Dimension scores

| Dimension | Baseline | After fixes |
|---|---|---|
| correctness | 0.472 | 0.733 |
| groundedness | 0.933 | 1.0 |
| completeness | 0.472 | 0.733 |
| refusal_appropriateness | 0.933 | 1.0 |

## 3. Judge scores per question (baseline -> fixed)

| ID | Tier | Adversarial | Baseline | Fixed | Judge note (baseline) |
|---|---|---|---|---|---|
| E01 | easy |  | 1.0 | 1.0 | All required facts present. |
| E02 | easy |  | 0.4 | 1.0 | Missing facts: ['$29']. |
| E03 | easy |  | 1.0 | 1.0 | All required facts present. |
| E04 | easy |  | 0.4 | 0.4 | Missing facts: ['United States', 'Canada', 'United Kingdom', 'Australia']. |
| E05 | easy |  | 1.0 | 1.0 | All required facts present. |
| E06 | easy |  | 1.0 | 1.0 | All required facts present. |
| E07 | easy |  | 1.0 | 1.0 | All required facts present. |
| E08 | easy |  | 1.0 | 1.0 | All required facts present. |
| E09 | easy |  | 1.0 | 1.0 | All required facts present. |
| E10 | easy |  | 0.4 | 1.0 | Missing facts: ['90 days']. |
| M01 | medium |  | 0.7 | 1.0 | Missing facts: ['$0.05']. |
| M02 | medium |  | 1.0 | 1.0 | All required facts present. |
| M03 | medium |  | 0.4 | 0.4 | Missing facts: ['950.40']. |
| M04 | medium |  | 0.4 | 0.7 | Missing facts: ['Growth', '$29']. |
| M05 | medium |  | 1.0 | 1.0 | All required facts present. |
| M06 | medium |  | 0.7 | 1.0 | Missing facts: ['5 user seats|5 seats']. |
| M07 | medium |  | 0.4 | 0.4 | Missing facts: ['1% conversion fee|1% currency conversion|1% conversion']. |
| M08 | medium |  | 0.8 | 1.0 | Missing facts: ['1000']. |
| M09 | medium |  | 0.7 | 0.7 | Missing facts: ['yes']. |
| M10 | medium |  | 0.7 | 1.0 | Missing facts: ['1-2 business days']. |
| H01 | hard | no_answer | 0.0 | 1.0 | Answered an unanswerable question instead of abstaining. |
| H02 | hard | misleading_phrasing | 0.7 | 1.0 | Missing facts: ['not refundable']. |
| H03 | hard | hallucination_bait | 1.0 | 1.0 | All required facts present. |
| H04 | hard |  | 0.4 | 0.4 | Missing facts: ['$24']. |
| H05 | hard |  | 0.4 | 0.4 | Missing facts: ['6.10']. |
| H06 | hard |  | 1.0 | 1.0 | All required facts present. |
| H07 | hard |  | 0.4 | 0.4 | Missing facts: ['278.40', '69.60']. |
| H08 | hard | no_answer | 0.0 | 1.0 | Answered an unanswerable question instead of abstaining. |
| H09 | hard |  | 0.4 | 0.4 | Missing facts: ['$65']. |
| H10 | hard |  | 0.4 | 1.0 | Missing facts: ['1 user seat', 'Growth']. |

## 4. Five failure diagnoses (baseline)

- **H01** - Prompt: the KB has no answer but the v1 prompt has no 'abstain if unsupported' rule, so the bot answered with an irrelevant fact (hallucination risk).
- **H08** - Prompt: the KB has no answer but the v1 prompt has no 'abstain if unsupported' rule, so the bot answered with an irrelevant fact (hallucination risk).
- **E02** - Retrieval: needed chunk(s) ['C02'] were not in the top-1 results (got ['C04']), so the generator never saw the required facts.
- **E04** - Generation: the right context was retrieved but the answer is incomplete/wrong (Missing facts: ['United States', 'Canada', 'United Kingdom', 'Australia'].) - answer was cut to a single sentence or needs arithmetic/multi-part reasoning.
- **E10** - Generation: the right context was retrieved but the answer is incomplete/wrong (Missing facts: ['90 days'].) - answer was cut to a single sentence or needs arithmetic/multi-part reasoning.

## 5. Implemented fixes and re-evaluation

| ID | Failure type | Fix | Before | After | Result |
|---|---|---|---|---|---|
| H01 | prompt | F2 prompt: v2 system prompt with abstain/false-premise rules | 0.0 | 1.0 | improved |
| H08 | prompt | F2 prompt: v2 system prompt with abstain/false-premise rules | 0.0 | 1.0 | improved |
| E02 | retrieval | F1 retrieval: top_k 1->3 + query expansion (synonyms) | 0.4 | 0.4 | not fixed |
| E04 | generation | F3 generation: answer budget 1->3 sentences / 'complete answer' instruction | 0.4 | 1.0 | improved |
| E10 | generation | F3 generation: answer budget 1->3 sentences / 'complete answer' instruction | 0.4 | 1.0 | improved |

**4/5 fixed** (requirement: at least 3).

## 6. Remaining weaknesses after fixes (lowest 5)

- E04 (0.4): Missing facts: ['United States', 'Canada', 'United Kingdom', 'Australia'].
- M03 (0.4): Missing facts: ['950.40'].
- M07 (0.4): Missing facts: ['1% conversion fee|1% currency conversion|1% conversion'].
- H04 (0.4): Missing facts: ['$24'].
- H05 (0.4): Missing facts: ['6.10'].

## 7. Regression test

`python regression_test.py` runs the full 30-question suite, writes a timestamped file to `results/` and prints PASS/FAIL.