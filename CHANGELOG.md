# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-18

First public release. Alpha: formats and interfaces may still change.

### Added

- Writer keys with scrypt-encrypted, atomically written keyrings (`0600`), public
  fingerprints, domain-scoped verification subkeys, rotation, and a persisted revocation flag.
- Tier-1 embedding over a hand-audited equivalence table (135 classes), with per-slot diff
  preview, interactive veto, and an anti-slop style guard.
- A single shared traversal for embedding and detection, so a verifier re-derives the same
  slots from a copied text; anchors are words the watermark cannot rewrite.
- Detection with variable-arity statistics, multiplicity correction across multiple keys,
  mandatory caveats, and no truth value (`bool(decision)` raises).
- Model-free sidecar verification from a published per-slot ledger.
- Tier-2 identity payload codec (24-bit writer id + 13-bit date + 8-bit CRC) with per-bit
  voting and a permissive-failure report.
- Detached provenance manifests: HMAC-SHA256 by default, Ed25519 through the `filigrana[sign]`
  extra; text differences are reported as edits, not forged signatures.
- A self-hosted registry format mapping writer ids to handles.
- An optional model-gated path (`gate.py`, `lm.py`, `lexical.py`) that keeps only
  near-indifferent substitution proposals.
- Capacity, attack, model-gate, and style-interference measurements with machine-readable
  copies in `bench/` and a reproduction guide in `docs/CAPACITY.md`.
- CLI commands: `key new`, `key show`, `embed`, `detect`, `table`, `publish`, `verify`,
  `registry`, and `audit`.

### Security

- Keyring writes go through a temporary file, `fsync`, and `os.replace`; a failed save leaves
  the previous keyring intact.
- Keyring files are validated before use: format, hex fields, field lengths, and a fingerprint
  that must match the decrypted secret. Permissive file modes produce a warning.
- The keyring refuses to run on non-POSIX platforms.
- Sidecars, manifests, and registries validate their format and JSON shape before use.
- No network I/O in detection or verification.

### Known limitations

- The mark survives verbatim or lightly edited republication. Full paraphrase and translation
  remove it (~0% measured).
- Short texts cannot carry enough independent choices; detection refuses to look confident
  below eight slots.
- Tier 2 needs book-length text; tier-1 attribution is the practical path at essay length.
- Windows is unsupported.
- The public-verification problem (anyone can verify, nobody can forge) is not solved here.
  See `docs/THREAT-MODEL.md`.

[0.1.0]: https://github.com/moah0911/filigrana/releases/tag/v0.1.0
