# ADR 001: Governance Layer

Roadmap status: future-facing note, not the primary repository description.

## Decision

Keep schema checks, data validation, cleaning, and manifest generation grouped clearly, while treating any larger control-plane framing as future documentation rather than a claim of a fully layered production system.

## Implemented

- `src/dc_reif/governance/`
- runtime manifest generation
- schema contract registry

## Planned

- stronger automated data contracts
- richer corruption taxonomies
