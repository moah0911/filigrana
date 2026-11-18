# Security policy

## Supported versions

`0.1.x` is the current alpha line. Security fixes land on the latest `0.1.x` release; older
pre-release snapshots are not supported.

## Reporting a vulnerability

Do not open a public issue for a security report. Use GitHub's private vulnerability reporting
for [`moah0911/filigrana`](https://github.com/moah0911/filigrana/security/advisories/new), or
email the maintainer listed on the repository profile. Include:

- the affected version and platform,
- a minimal reproduction (text and key material can be synthetic),
- what an attacker gains, and
- whether the issue is in the library, the CLI, or a documented format.

You should receive an acknowledgement within a few days. Please give us a reasonable window to
ship a fix before publishing details.

## What counts as a security issue here

`filigrana` deliberately identifies a person, so the security boundary is unusual:

- **Key disclosure or forgery.** Anyone who reads a keyring can both verify that writer and
  plant their mark on prose they never wrote. Weak keyring permissions, non-atomic writes,
  passphrase handling, and subkey scoping are in scope.
- **False attribution.** A statistical or format bug that makes a foreign key look like the
  writer's key, or that turns an edit into a forged-signature report, is in scope.
- **Verification bypass.** A sidecar, manifest, or registry that verifies when it should not.
- **Network or filesystem surprises.** Detection and verification are documented as offline
  and local. Unexpected network I/O, path traversal, or unsafe temporary-file handling is in
  scope.

Out of scope: the mark failing against paraphrase or translation (documented, measured, and
inherent to in-band word-choice watermarking); a key holder lying about their own key; and the
absence of a hosted verification service, which is a design decision, not a vulnerability.

## Handling keys

- The keyring is written with mode `0600` using a temporary file, `fsync`, and `os.replace`.
- A permissive keyring mode produces a warning on load.
- The passphrase can come from `FILIGRANA_PASSPHRASE` for non-interactive use. Prefer the
  prompt on a shared machine; an environment variable is visible to the same user's processes.
- Publish the fingerprint, never the secret. Hand a publisher a domain-scoped subkey
  (`WriterKey.for_domain`) rather than the master key.
- Revocation is recorded in the key file, but it cannot retract text that was already
  published or a key that has already leaked.

## Formats and compatibility

The on-disk formats are versioned (`filigrana-keyring/1`, `filigrana-manifest/1`,
`filigrana-registry/1`, `filigrana-sidecar/2`). Verification refuses a format it does not
recognise rather than guessing. Cryptographic domain-separation strings retain their
pre-release `inkmark/1` values on purpose; changing them would invalidate every existing
fingerprint and word choice.

## More detail

The threat model, including what a detection result does and does not mean, is in
[`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md).
