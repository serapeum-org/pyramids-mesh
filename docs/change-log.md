## 0.1.0 (2026-10-13)

### Feat

- **scaffold**: initial repository layout — `src/pyramids_mesh` package with the
  `import pyramids`-first contract, pytest suite, pixi environments (`default`, `dev`, `docs`,
  `py311`-`py314`), MkDocs Material + mkdocstrings docs site, pre-commit hooks, and six CI
  workflows built on the shared `serapeum-org/github-actions`: tests, lint (pre-commit + src
  doctests + gitleaks secret scan), pip-audit, MkDocs deploy, GitHub release, PyPI release.

No mesh API is published yet. The package exports `__version__` only.
