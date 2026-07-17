# Contributing

- Keep changes narrowly scoped and avoid mixing product work with cleanup.
- Run targeted tests, the full test suite, and `llm-modelbench selftest` before
  requesting release review.
- Keep scorer changes pure and regression-tested; do not add cosmetic metrics.
- Add tasks with honest levels and difficulties, and never add public fixtures
  that violate [PRIVACY_FIXTURES.md](PRIVACY_FIXTURES.md).
- Do not run models, alter fixtures, or change release scope without explicit
  approval.
- Record release notes in [CHANGELOG.md](../CHANGELOG.md).
- Make local commits first. Tags and pushes happen only after review and
  acceptance.
