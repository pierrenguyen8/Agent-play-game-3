# Migration manifest

## Migrated as clean production concepts
- Generic actions
- Intent
- WorldState
- GameAdapter seam
- PlanetAdapter
- Planet keyboard/input state abstraction
- Screen capture abstraction
- Planet perception contract
- Planet StateBuilder
- Rule-based high-level policy
- SkillManager
- Runtime façade
- Gameplay data contracts
- Held-state transition decoder

## Preserved as external assets by migration script
- Raw demonstrations (optional)
- Final temporal processed data (optional)
- Best player/enemy/platform weights
- Legacy final-temporal V1 checkpoint + evaluation report

## Intentionally left behind
- old low-level neural architecture implementations
- V1/V2/final/structured source variants
- old losses/trainers/samplers
- smoke scripts and one-off diagnostics
- generated training runs except selected best artifacts

Reason: the new low-level layer is reuse-first and must be selected by benchmark, not inherited by accident.
