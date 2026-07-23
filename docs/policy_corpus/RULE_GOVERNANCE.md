# Rule Governance Policy
Version: 2026.1

## Version History
- RULE_SET_V1 (2025-01-15): Initial conservative scorecard
- RULE_SET_V2 (2026-05-03): ML-assisted scoring path

## Change Process
1. New rule versions are immutable dataclasses in engine/rule_sets.py
2. Each version must reference the change request and approval date
3. Active version is set via ACTIVE_RULE_SET in engine/rule_sets.py
4. A/B testing between rule versions uses ml/governance/ab_test.py

## Approval Matrix
- Minor threshold adjustments: Credit Risk Committee
- New rule versions: Credit Risk Committee + Compliance
- Weight changes > 10%: Board approval required
