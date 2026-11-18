"""Writer-facing CLI.

`embed` shows its diff and refuses to write unless you name an output file, because the
measured quality cost of automatic synonym swapping is real and a writer, not a tool,
should own it. `detect` prints a p-value and its caveats and never a verdict word.
"""


import argparse
import getpass
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__
from .canon import canonical_form
from .codec import decode_identity, encode_identity, read_payload
from .decisions import normal_tail
from .detect import scan, scan_against
from .embed import embed
from .keyring import WriterKey
from .ledger import walk, word_count
from .providers import default_table, normalise_tier
from .registry import Registry, entry_for
from .slop import is_slop


class CLIError(Exception):
    """A user-facing failure: print it once and exit 2, with no traceback."""


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CLIError(f"no such file: {path}") from None
    except IsADirectoryError:
        raise CLIError(f"{path} is a directory, not a file") from None
    except PermissionError:
        raise CLIError(f"cannot read {path}: permission denied") from None
    except UnicodeDecodeError as exc:
        raise CLIError(f"{path} is not valid UTF-8 (byte {exc.start})") from None
    except OSError as exc:
        raise CLIError(f"cannot read {path}: {exc.strerror or exc}") from None


def _load_json(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLIError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from None


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CLIError(f"cannot write {path}: {exc.strerror or exc}") from None


def _tier(value: str) -> str:
    try:
        return normalise_tier(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _payload_bits(value: str) -> int:
    try:
        bits = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--payload-bits must be an integer") from None
    if not 1 <= bits <= 64:
        raise argparse.ArgumentTypeError("--payload-bits must be between 1 and 64")
    return bits


def _date_arg(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"date must be YYYY-MM-DD, not {value!r}") from None
    return value


def _parse_payload(value: str) -> tuple[str, date | None]:
    handle, sep, published = value.partition(",")
    if not handle:
        raise argparse.ArgumentTypeError("--payload needs a non-empty HANDLE")
    if sep and published:
        try:
            return handle, date.fromisoformat(published)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--payload date must be YYYY-MM-DD, not {published!r}"
            ) from None
    return handle, None


def _passphrase(confirm: bool = False) -> str:
    from_env = os.environ.get("FILIGRANA_PASSPHRASE")
    if from_env:
        return from_env
    try:
        first = getpass.getpass("filigrana passphrase: ")
    except EOFError:
        # No TTY (cron, CI, a piped script): a traceback tells nobody anything.
        raise CLIError(
            "a passphrase is required. Run interactively, or set FILIGRANA_PASSPHRASE."
        ) from None
    if confirm:
        try:
            second = getpass.getpass("again: ")
        except EOFError:
            raise CLIError(
                "confirmation prompt lost its stdin; run interactively, "
                "or set FILIGRANA_PASSPHRASE."
            ) from None
        if first != second:
            raise CLIError("passphrases did not match")
    return first


def _load_key(path: Path) -> WriterKey:
    if not path.exists():
        raise CLIError(f"no keyring at {path}; run `filigrana key new --out {path}`")
    if not path.is_file():
        raise CLIError(f"{path} is not a file")
    try:
        return WriterKey.load(path, _passphrase())
    except ValueError as exc:
        raise CLIError(str(exc)) from None


def cmd_key_new(args: argparse.Namespace) -> int:
    out = _path(args.out)
    key = WriterKey.generate(note=args.note or "")
    try:
        key.save(out, _passphrase(confirm=True))
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    print(f"key id       {key.key_id}")
    print(f"fingerprint  {key.fingerprint}   <- publish this; it is how your mark resolves to you")
    print(f"saved        {out} (0600, scrypt-encrypted)")
    print("\nKeep the keyring private. Anyone who reads it can both verify you and forge you.")
    return 0


def cmd_key_show(args: argparse.Namespace) -> int:
    key = _load_key(_path(args.key))
    print(f"key id       {key.key_id}")
    print(f"fingerprint  {key.fingerprint}")
    print(f"writer id    0x{int(key.fingerprint[:6], 16):x}")
    print(f"created      {key.created}  epoch {key.epoch}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    source = _path(args.file)
    text = _read_text(source)
    key = _load_key(_path(args.key))
    table = default_table(args.tier)

    payload, n_bits = None, 0
    if args.payload:
        _, published = args.payload
        stamp = published or date.today()
        payload, n_bits = encode_identity(key.fingerprint, stamp)
        print(f"payload      0x{payload:x} ({n_bits} bits): id 0x{payload >> 21:x} + {stamp}\n")

    def veto(slot, wanted: int) -> bool:
        """True keeps the writer's own word. Asked per slot, never in bulk."""

        if not args.interactive:
            return False
        original = text[slot.start : slot.end]
        members = table.classes[slot.class_id].members
        prompt = f"key wants {members[wanted]!r} for {original!r} [{'/'.join(members)}] (y/N) "
        try:
            answer = input(prompt)
        except EOFError:
            # `filigrana embed --interactive < piped-script` must not traceback mid-veto.
            raise CLIError(
                "--interactive needs a terminal to ask you per slot; "
                "re-run without --interactive or with a real stdin."
            ) from None
        return answer.strip().lower() in ("y", "yes")

    result = embed(
        text,
        key,
        table=table,
        payload=payload,
        n_bits=n_bits,
        veto=veto,
        disprefer=is_slop if args.anti_slop else None,
    )

    print(result.summary())
    print(f"table        {table.class_count} classes, tier {args.tier}\n")
    if result.edits:
        print("proposed edits")
        for edit in result.edits:
            print(f"  {edit.render()}")
    else:
        print("no edits needed: every slot already landed where your key points")

    if args.out is None:
        print("\n(dry run -- nothing written. add --out FILE to save.)")
        if args.interactive:
            print("note: interactive vetoes above were previews only; re-run with --out to keep them.")
        return 0

    out = _path(args.out)
    _write_text(out, result.text)
    if args.sidecar:
        sidecar_path = _path(args.sidecar)
        _write_text(
            sidecar_path,
            json.dumps(
                {**result.sidecar(), "key_id": key.key_id, "words": result.words}, indent=1
            ),
        )
    print("\nwrote " + str(out) + (f" and {args.sidecar}" if args.sidecar else ""))
    print(
        "re-detect before publishing: filigrana detect "
        + str(out)
        + " --key "
        + str(_path(args.key))
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    text = _read_text(_path(args.file))
    keys = [_load_key(_path(p)) for p in args.key]
    table = default_table(args.tier)
    sidecar = _load_json(_path(args.sidecar)) if args.sidecar else None

    if len(keys) == 1:
        decision = scan(text, keys[0], table=table, sidecar=sidecar)
    else:
        decision = scan_against(text, keys, table=table, sidecar=sidecar)

    print(decision.render())

    if args.payload_bits:
        best = max(keys, key=lambda k: scan(text, k, table=table).z)
        read = read_payload(text, best, args.payload_bits, table=table)
        _report_payload(read, args.payload_bits)
    return 0


def _report_payload(read, n_bits: int) -> None:
    """Print a tier-2 payload read. Separated from cmd_detect so both outcomes are
    testable without the ~60,000 words the 45-bit wire format genuinely needs."""
    value = read.payload
    if value is None:
        print(
            f"payload      undecoded: {read.unresolved} of {n_bits} bits tied, "
            f"{read.inconclusive} slots carried no information"
        )
        print("             this is the tier-2 discovery path failing the way it should: "
              "the text is too short for the bits, not evidence against the writer.")
    else:
        writer_id, published, crc_ok = decode_identity(value, n_bits)
        print(f"payload      0x{value:x} -> writer id 0x{writer_id:x}, published {published}")
        print(f"checksum     {'ok' if crc_ok else 'MISMATCH - decode is not trustworthy'}")
        print(f"weakest bit  p = {read.weakest:.3g}")


def cmd_table(args: argparse.Namespace) -> int:
    table = default_table(args.tier)
    print(f"tier {args.tier}: {table.class_count} classes, {table.member_count} surfaces")
    problems = table.audit()
    if problems:
        print("surfaces claimed by more than one class (both guards must disambiguate):")
        for line in problems:
            print(f"  {line}")
    if args.slots_from:
        text = _read_text(_path(args.slots_from))
        entries = walk(text, WriterKey(secret=b"\0" * 32, key_id="probe", created="", year=0), table)
        by_tier: dict[str, int] = {}
        for e in entries:
            by_tier[e.tier] = by_tier.get(e.tier, 0) + 1
        words = word_count(text)
        print(
            f"{words} words -> {len(entries)} slots "
            f"({1000 * len(entries) / max(words, 1):.1f} per 1000 words) {by_tier}"
        )
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from .manifest import ed25519_available, publish

    source = _path(args.file)
    text = _read_text(source)
    key = _load_key(_path(args.key))
    verify_key = key.for_domain(args.domain).secret if args.domain else None
    manifest = publish(
        canonical_form(text),
        key,
        handle=args.handle,
        published=args.published,
        url=args.url or "",
        license=args.license or "",
        verify_key=verify_key,
    )
    out = _path(args.out) if args.out else _path(str(source) + ".filigrana.json")
    _write_text(out, manifest.to_json())
    print(f"manifest     {out}")
    print(f"text sha256  {manifest.digest[:32]}...  (over the canonical form, not the file)")
    print(f"scheme       {manifest.scheme}")
    if args.domain:
        print(f"scoped to    {args.domain} -- hand that subkey to this publisher only")
    if not ed25519_available():
        print(
            "\n  ! No public signature: the attestation key is also the signing key, so you\n"
            "    must nominate a verifier rather than let the world check. Install\n"
            "    'filigrana[sign]' for Ed25519. Not a C2PA manifest and not a substitute for one."
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .manifest import Manifest, verify

    text = _read_text(_path(args.file))
    manifest_path = _path(args.manifest)
    try:
        manifest = Manifest.from_json(_read_text(manifest_path))
    except ValueError as exc:
        raise CLIError(f"{manifest_path}: {exc}") from None
    key = _load_key(_path(args.key))
    check = verify(
        canonical_form(text),
        manifest,
        key.for_domain(args.domain).secret if args.domain else key.secret,
    )
    print(f"text matches      {check['text_matches']}")
    print(f"attestation ok    {check['attestation_matches']}")
    print(f"scheme            {check['scheme']}")
    for field in ("handle", "published", "url", "license", "fingerprint", "minted"):
        print(f"{field:<16}  {check['claims'].get(field, '')}")
    if check["text_matches"] and check["attestation_matches"]:
        print("\nThis copy is the one that was published under this key.")
    elif not check["text_matches"]:
        print(
            "\nThe text differs from the published copy. That is an edit, not a broken"
            "\nsignature -- report them separately or provenance tools get trusted and wrong."
        )
    else:
        print("\nAttestation does not verify under this key.")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    path = _path(args.file)
    if path.exists():
        try:
            registry = Registry.from_json(_read_text(path))
        except ValueError as exc:
            raise CLIError(f"{path}: {exc}") from None
    else:
        registry = Registry()
    if args.add:
        key = _load_key(_path(args.key))
        entry = entry_for(key, args.add, args.url or "")
        try:
            registry.add(entry)
        except ValueError as exc:
            raise CLIError(str(exc)) from None
        _write_text(path, registry.to_json())
        print(f"added {entry.writer_id} -> {entry.handle} ({path})")
        print("Host this file yourself, ideally at /.well-known/filigrana.json")
    else:
        print(registry.to_json())
        print(f"\n{len(registry)} entries")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import math

    text = _read_text(_path(args.file))
    key = _load_key(_path(args.key))
    table = default_table(args.tier)
    entries = walk(text, key, table, mode="detect")
    words = word_count(text) or 1
    rho = 1000 * len(entries) / words
    tiers: dict[str, int] = {}
    for e in entries:
        tiers[e.tier] = tiers.get(e.tier, 0) + 1
    variance = sum((1 / e.n) * (1 - 1 / e.n) for e in entries) or 1e-9
    z_full = (len(entries) - sum(1 / e.n for e in entries)) / math.sqrt(variance)
    needed = 4.754**2 / max(rho / 1000, 1e-9)
    print(f"{words} words, {len(entries)} choices ({rho:.1f} per 1000 words) {tiers}")
    print(f"best possible z at full strength: {z_full:.2f}  (p = {normal_tail(z_full):.3g})")
    print(f"words needed for p<1e-6 at this density: {needed:,.0f}")
    for survive, label in ((1.0, "verbatim copy"), (0.75, "25% of slots edited away"),
                           (0.5, "half the text quoted"), (0.25, "a quarter of slots left")):
        reduced = z_full * math.sqrt(survive)
        print(f"  {label:<26} z -> {reduced:>6.2f}   p -> {normal_tail(reduced):.3g}")
    print(
        "\n! This is a prediction of the best case, assuming you apply every demand. Every"
        "\n  vetoed slot and every locale-normalising edit at the publisher moves you down."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="filigrana", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"filigrana {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    key = sub.add_parser("key", help="manage your writer key")
    key_sub = key.add_subparsers(dest="keycmd", required=True)
    new = key_sub.add_parser("new", help="create a keyring")
    new.add_argument("--out", default="~/.filigrana/keyring.json")
    new.add_argument("--note", default="")
    new.set_defaults(func=cmd_key_new)
    show = key_sub.add_parser("show", help="print the public fingerprint")
    show.add_argument("--key", default="~/.filigrana/keyring.json")
    show.set_defaults(func=cmd_key_show)

    em = sub.add_parser("embed", help="mark prose you wrote")
    em.add_argument("file")
    em.add_argument("--key", default="~/.filigrana/keyring.json")
    em.add_argument(
        "--tier", type=_tier, default="ab", help="a, b, c, ab, ac, bc, abc, or all"
    )
    em.add_argument(
        "--payload", type=_parse_payload, metavar="HANDLE,YYYY-MM-DD", help="tier-2: encode identity+date"
    )
    em.add_argument("--out", default=None, help="omit to preview only")
    em.add_argument("--sidecar", default=None, help="also write the slot ledger")
    em.add_argument("--interactive", action="store_true", help="veto each edit")
    em.add_argument(
        "--anti-slop",
        action="store_true",
        help="keep your own word where the key would push it toward machine-sounding filler. "
        "Costs statistical strength; `filigrana audit` prices it.",
    )
    em.set_defaults(func=cmd_embed)

    de = sub.add_parser("detect", help="test a suspect copy against a key")
    de.add_argument("file")
    de.add_argument("--key", action="append", required=True, help="repeatable for registry search")
    de.add_argument("--tier", type=_tier, default="ab")
    de.add_argument("--sidecar", default=None)
    de.add_argument("--payload-bits", type=_payload_bits, default=0)
    de.set_defaults(func=cmd_detect)

    ta = sub.add_parser("table", help="inspect the equivalence table")
    ta.add_argument("--tier", type=_tier, default="ab")
    ta.add_argument("--slots-from", default=None, metavar="FILE")
    ta.set_defaults(func=cmd_table)

    pub = sub.add_parser("publish", help="write a signed provenance manifest next to the in-band mark")
    pub.add_argument("file")
    pub.add_argument("--key", default="~/.filigrana/keyring.json")
    pub.add_argument("--handle", required=True)
    pub.add_argument("--published", type=_date_arg, default=str(date.today()))
    pub.add_argument("--url", default="")
    pub.add_argument("--license", default="")
    pub.add_argument("--domain", default=None, help="scope the attestation to one publisher")
    pub.add_argument("--out", default=None)
    pub.set_defaults(func=cmd_publish)

    ver = sub.add_parser("verify", help="check a copy against a manifest")
    ver.add_argument("file")
    ver.add_argument("--manifest", required=True)
    ver.add_argument("--key", default="~/.filigrana/keyring.json")
    ver.add_argument("--domain", default=None)
    ver.set_defaults(func=cmd_verify)

    reg = sub.add_parser("registry", help="the self-hosted id -> handle file for tier 2")
    reg.add_argument("--file", default="filigrana-registry.json")
    reg.add_argument("--add", default=None, metavar="HANDLE")
    reg.add_argument("--url", default="")
    reg.add_argument("--key", default="~/.filigrana/keyring.json")
    reg.set_defaults(func=cmd_registry)

    au = sub.add_parser("audit", help="what this article can support before you publish it")
    au.add_argument("file")
    au.add_argument("--key", default="~/.filigrana/keyring.json")
    au.add_argument("--tier", type=_tier, default="ab")
    au.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CLIError as exc:
        print(f"filigrana: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("filigrana: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `filigrana table | head` closes the pipe early; that is not an error.
        return 0
    except (ValueError, OSError) as exc:
        print(f"filigrana: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
