# RC16 exact-scoring punctuation patch

RC16 is a score-contract-only follow-up to the public-readiness RC15 baseline.

## Change

`score_exact` still compares the complete visible response with the complete expected
answer after thinking-block removal, case folding, and whitespace normalization. It now
also ignores harmless trailing sentence marks: `.`, `!`, `?`, and `…`.

The change does **not**:

- accept an expected phrase found inside prose;
- accept mixed correct and incorrect answers;
- remove commas, colons, or semicolons;
- alter internal punctuation;
- change `score_exact_code` or any other scorer.

## Required regression gates

- correct answer plus terminal sentence punctuation scores 100;
- wrong answer scores 0;
- a correct phrase mentioned before a wrong final answer scores 0;
- comma, colon, and semicolon endings remain exact-significant;
- the complete test suite, release checks, clean-wheel install, and self-test pass.
