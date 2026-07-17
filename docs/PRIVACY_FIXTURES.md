# Fixture Content Policy

Public benchmark fixtures (tasks, prompts, documents, and expected answers)
must be synthetic and independent of any real person's circumstances. Use
fictional entities, generic technical scenarios, or clearly invented data.

## What synthetic means here

A fixture is synthetic when a reader has no way to trace it back to a real
person, organization, or event. In practice:

- Use invented names, invented companies, and invented document content.
- Do not adapt or lightly reword real personal, medical, financial, legal,
  immigration, or administrative material, even anonymized. Anonymization is
  not sufficient; the scenario itself must be invented.
- Do not use real case numbers, real form references, or real institutional
  wording lifted from an actual document.
- Generic technical or business scenarios (IT support tickets, expense
  claims, fictional company policies, made-up product documentation) are
  the preferred source of variety and difficulty.

## Retrieval fixtures specifically

Retrieval cases should use fictional internal policies and services.
Distractors should be semantically plausible without depending on private
facts or one-off "magic keywords" that only make sense with outside context.
Preserve target IDs and diagnostics provenance, but do not emit full queries
or document text in public case diagnostics by default.

## Review before adding

Any new public task or fixture should be reviewable by someone with no
context on how it was written: does it read as generic and invented, or does
it read as derived from something specific? If in doubt, treat it as
requiring review before it's added.

## Audit and review evidence

Private review material, raw operator evidence, and unreviewed audit notes
belong outside the public repository, not archived inside it.
