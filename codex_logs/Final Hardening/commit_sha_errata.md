# Commit SHA Errata

Generated during Final Hardening H4 from Git-resolved commit identities. Historical audit `.log` files were not rewritten.

Some generated handoffs and overnight summaries expanded correct short SHAs to incorrect full SHAs. The canonical values below are from `git rev-parse` and `git log`, not manual abbreviation expansion.

Notable corrected values:
- Stage 3B actual: `dabbe6f6265d522897b7706aeae9d8dae54b2418`
- Stage 4B actual: `1ab934b0cacd3f0a8803a1a7f84d3f4ed922fe38`

| stage | canonical full SHA | parent | subject |
| --- | --- | --- | --- |
| Stage 1A implementation | `f9836f47b56ef9f0d01cc42bce7779fe8a6c6d28` | `555028e925e7f35c89b7ec7f15432bcd80aff811` | Implement Stage 1A judge policy selection |
| Stage 1A corrective | `d2d38c5237dba1ecb563cadf2b6161d0e42feadc` | `f9836f47b56ef9f0d01cc42bce7779fe8a6c6d28` | Correct Stage 1A judge selection edge cases |
| Stage 1B implementation | `f850d10cf24702310402e32a4b9daeb6ed5dcab1` | `d2d38c5237dba1ecb563cadf2b6161d0e42feadc` | Implement Stage 1B judge qualification protocol |
| Stage 1B corrective | `3b4a550397f12074c9d87d923b932d06fcc69d18` | `f850d10cf24702310402e32a4b9daeb6ed5dcab1` | Correct Stage 1B qualification contracts |
| Stage 1B finalization | `a0a6eca09d52377eb8dac1ed9d934a1c1721b4bb` | `3b4a550397f12074c9d87d923b932d06fcc69d18` | Finalize Stage 1B qualification contracts |
| Stage 1C implementation | `a13f349dfe612f3c3e6a323f89ac460ae0ff754b` | `a0a6eca09d52377eb8dac1ed9d934a1c1721b4bb` | Implement Stage 1C independent judge roles |
| Stage 1C corrective | `f29704ccbc9cfec04c5ecf6b28c1564a511a0925` | `a13f349dfe612f3c3e6a323f89ac460ae0ff754b` | Correct Stage 1C independent judge integration |
| Stage 1D implementation | `2d0a166c752c23f5be6df82f0bfbb706e05790af` | `f29704ccbc9cfec04c5ecf6b28c1564a511a0925` | Close Stage 1D judge provenance integration |
| Stage 1D corrective | `d7a1ba09f476d7f0c7b18014e7385e4fd8f590c3` | `2d0a166c752c23f5be6df82f0bfbb706e05790af` | Correct Stage 1D judge failure provenance |
| Stage 1D amendment | `92cb80df1fa9f488b2f770f62ebee7eb485774c9` | `d7a1ba09f476d7f0c7b18014e7385e4fd8f590c3` | Amend Stage 1D structural fallback orchestration |
| Stage 2A implementation | `38f1b8acd301597214b7bf5bc4a25becda115d50` | `92cb80df1fa9f488b2f770f62ebee7eb485774c9` | Implement Stage 2A recovery reconciliation |
| Stage 2A corrective | `420b84cf06aa2dd75ddbaf20752b9b39be2d063d` | `38f1b8acd301597214b7bf5bc4a25becda115d50` | Tighten Stage 2A recovery attribution |
| Stage 2B implementation | `065262c5d54fbae4b33ab4ee8786fed8367bd5f7` | `420b84cf06aa2dd75ddbaf20752b9b39be2d063d` | Implement Stage 2B recovery outcomes |
| Stage 2B corrective | `cbe92493bcf53b7779c249c0956763136bbb8719` | `065262c5d54fbae4b33ab4ee8786fed8367bd5f7` | Correct Stage 2B recovery provenance |
| Stage 2B corrective | `f06512bb1133a1a6926d26efe16a0d78868f9bc6` | `cbe92493bcf53b7779c249c0956763136bbb8719` | Correct Stage 2B final recovery selection |
| Stage 2B corrective | `35fdac0b437f74b2da83d564fbf50bfa14c5ea0e` | `f06512bb1133a1a6926d26efe16a0d78868f9bc6` | Fail closed on invalid Stage 2B final evidence |
| Stage 3A implementation | `11d787a0eeb0dccc2a2d2cdc11cd8f7261bdc3c7` | `35fdac0b437f74b2da83d564fbf50bfa14c5ea0e` | Implement Stage 3A supersession graph safety |
| Stage 3A corrective | `66e5265652d01e927fa6d846ab689c520b48fd3b` | `11d787a0eeb0dccc2a2d2cdc11cd8f7261bdc3c7` | Harden Stage 3A supersession validation |
| Stage 3B implementation | `dabbe6f6265d522897b7706aeae9d8dae54b2418` | `66e5265652d01e927fa6d846ab689c520b48fd3b` | Implement Stage 3B supersession resolution |
| Stage 4A implementation | `aeeb6e0efd710555b8b1a1a2dcc4899df20f6fae` | `dabbe6f6265d522897b7706aeae9d8dae54b2418` | Implement Stage 4A campaign config schema |
| Stage 4B implementation | `1ab934b0cacd3f0a8803a1a7f84d3f4ed922fe38` | `aeeb6e0efd710555b8b1a1a2dcc4899df20f6fae` | Implement Stage 4B config execution guards |
| Stage 5 implementation | `464ed3f68ffbb070fb7102509a97b4a6cffcb20c` | `1ab934b0cacd3f0a8803a1a7f84d3f4ed922fe38` | Implement Stage 5 readiness integration tests |
| Stage 6 implementation | `4bdf2e0252555376f6444692f56e2e4f84649467` | `464ed3f68ffbb070fb7102509a97b4a6cffcb20c` | Document acceptance controls UX |
| Stage 7 implementation | `1169ac2c07f486f78914e7ecf7530d5470de7f74` | `4bdf2e0252555376f6444692f56e2e4f84649467` | Complete Stage 7 offline validation |
| H1 final hardening | `9ccdac3abbecc7ae20eac4fb6d150e7c4c4c1dfa` | `1169ac2c07f486f78914e7ecf7530d5470de7f74` | Harden config execution freeze ordering |
| H2 final hardening | `eaa250133a5d1895154c9143277c56eda6390300` | `9ccdac3abbecc7ae20eac4fb6d150e7c4c4c1dfa` | Anchor applied supersession replacements |
| H3 final hardening | `41d54cf2def2b868284be2bc3cd7e65d578646f9` | `eaa250133a5d1895154c9143277c56eda6390300` | Validate packaged supersession and config evidence |

