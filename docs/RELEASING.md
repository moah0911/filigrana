# Releasing

The release path is deliberately manual at the final step. CI builds and attaches artifacts to
a GitHub release; a human with a PyPI token uploads to the index. No publish credential lives
in this repository or in GitHub Actions.

## Before you tag

1. **Version.** `pyproject.toml` reads the version dynamically from
   `src/filigrana/__init__.py`. Bump `__version__` there once, and confirm the tag matches
   (`v0.1.0` for `0.1.0`).
2. **Formats.** Do not rename the frozen `inkmark/1` cryptographic domain strings in
   `keyring.py`, `keystream.py`, or `integrations/synthid.py`. They are derivation contracts,
   not branding. A rename silently invalidates every fingerprint and keyed word choice.
3. **Numbers.** Any number quoted in `README.md` or `docs/CAPACITY.md` must match the JSON in
   `bench/`. Re-run the relevant tool if the table changed; do not hand-edit a measurement.
4. **Docs.** Update `CHANGELOG.md`, check `SECURITY.md`, and run the example workflow in
   `examples/README.md` if the CLI output format changed.

## Verify the tree

```console
uv sync --locked --dev --extra sign --extra lexical
uv run ruff check src/ tools/ tests/
uv run mypy
uv run pytest --cov=filigrana --cov-report=term-missing --cov-fail-under=85
```

The default CI matrix runs Ubuntu 3.11–3.13 and macOS 3.12. The optional-extras job installs
`sign` and `lexical`; the `lm` extra is left out of CI on cost and is exercised by the
measurement tools.

## Build and test the distributions

```console
rm -rf dist build
uv run python -m build
uv run twine check dist/*
```

Install the wheel into a clean virtual environment, outside the source tree, and run a fresh
end-to-end flow:

```console
python -m venv /tmp/filigrana-release-venv
/tmp/filigrana-release-venv/bin/pip install dist/filigrana-0.1.0-py3-none-any.whl
/tmp/filigrana-release-venv/bin/filigrana --version
/tmp/filigrana-release-venv/bin/filigrana table --tier all
```

Then, with a throwaway keyring:

```console
export FILIGRANA_PASSPHRASE='release test passphrase'
/tmp/filigrana-release-venv/bin/filigrana key new --out /tmp/release-keyring.json
/tmp/filigrana-release-venv/bin/filigrana embed README.md \
    --key /tmp/release-keyring.json --out /tmp/readme.marked.md
/tmp/filigrana-release-venv/bin/filigrana detect /tmp/readme.marked.md \
    --key /tmp/release-keyring.json
```

Confirm the detect output contains a statistic and caveats, and no verdict.

## Tag and GitHub release

```console
git push origin master
git tag -a v0.1.0 -m "filigrana 0.1.0"
git push origin v0.1.0
```

Pushing the tag runs the `release` job in `.github/workflows/ci.yml`, which rebuilds the sdist
and wheel, runs `twine check`, and attaches both files to the GitHub release with generated
notes. Confirm the workflow passed before uploading to PyPI.

## PyPI

PyPI upload is manual from a release workstation. The token belongs in `~/.pypirc` (mode
`0600`) or in `TWINE_PASSWORD`; never in the repository, never in shell history, never in a
GitHub secret.

```console
uv run twine upload dist/*
```

Then verify the public index in a fresh environment:

```console
python -m venv /tmp/filigrana-pypi-venv
/tmp/filigrana-pypi-venv/bin/pip install filigrana==0.1.0
/tmp/filigrana-pypi-venv/bin/filigrana --version
```

If the upload fails, fix the metadata and rebuild before retrying. PyPI does not allow
re-uploading a version, even if the file was deleted; a bad upload means a new patch version.
