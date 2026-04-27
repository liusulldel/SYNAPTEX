# Release Checklist

Use this checklist before tagging a public SYNAPTEX release.

- Run `python -m compileall -q synaptex`.
- Run `python -m pytest`.
- Run `ruff check synaptex tests examples`.
- Run `python -m build`.
- Run `python -m twine check dist/*`.
- Confirm `LICENSE` is plain MIT and matches package metadata.
- Confirm no `.env`, credentials, caches, build artifacts, or private data are staged.
- Confirm README limitations still match current implementation.
