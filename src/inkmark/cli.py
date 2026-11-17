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

from . import __version__
from .canon import canonical_form
from .codec import decode_identity, encode_identity, read_payload
from .decisions import normal_tail
from .detect import scan, scan_against
from .embed import embed
from .keyring import WriterKey
from .ledger import walk, word_count
from .providers import default_table
from .registry import Registry, entry_for
from .slop import is_slop


def _passphrase(confirm: bool = False) -> str:
    from_env = os.environ.get("INKMARK_PASSPHRASE")
    if from_env:
        return from_env
    try:
        first = getpass.getpass("inkmark passphrase: ")
    except EOFError:
        # No TTY (cron, CI, a piped script): a traceback tells nobody anything.
        raise SystemExit(
            "inkmark: a passphrase is required. Run interactively, or set INKMARK_PASSPHRASE."
        ) from None
    if confirm:
        second = getpass.getpass("again: ")
        if first != second:
            raise SystemExit("passphrases did not match")
    return first


def _load_key(path: Path) -> WriterKey:
    if not path.exists():
        raise SystemExit(f"no keyring at {path}; run `inkmark key new --out {path}`")
    try:
        return WriterKey.load(path, _passphrase())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def cmd_key_new(args: argparse.Namespace) -> int:
    key = WriterKey.generate(note=args.note or "")
    key.save(Path(args.out), _passphrase(confirm=True))
    print(f"key id       {key.key_id}")
    print(f"fingerprint  {key.fingerprint}   <- publish this; it is how your mark resolves to you")
    print(f"saved        {args.out} (0600, scrypt-encrypted)")
    print("\nKeep the keyring private. Anyone who reads it can both verify you and forge you.")
    return 0


def cmd_key_show(args: argparse.Namespace) -> int:
    key = _load_key(Path(args.key))
    print(f"key id       {key.key_id}")
    print(f"fingerprint  {key.fingerprint}")
    print(f"writer id    0x{int(key.fingerprint[:6], 16):x}")
    print(f"created      {key.created}  epoch {key.epoch}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    key = _load_key(Path(args.key))
    table = default_table(args.tier)

    payload, n_bits = None, 0
    if args.payload:
        handle, _, published = args.payload.partition(",")
        del handle  # the writer id comes from the key fingerprint; the handle is CLI syntax only
        stamp = date.fromisoformat(published) if published else date.today()
        payload, n_bits = encode_identity(key.fingerprint, stamp)
        print(f"payload      0x{payload:x} ({n_bits} bits): id 0x{payload >> 21:x} + {stamp}\n")

    def veto(slot, wanted: int) -> bool:
        """True keeps the writer's own word. Asked per slot, never in bulk."""

        if not args.interactive:
            return False
        source = text[slot.start : slot.end]
        members = table.classes[slot.class_id].members
        answer = input(f"key wants {members[wanted]!r} for {source!r} [{'/'.join(members)}] (y/N) ")
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

    out = Path(args.out)
    out.write_text(result.text, encoding="utf-8")
    if args.sidecar:
        Path(args.sidecar).write_text(
            json.dumps({**result.sidecar(), "key_id": key.key_id, "words": result.words}, indent=1),
            encoding="utf-8",
        )
    print("\nwrote " + str(out) + (f" and {args.sidecar}" if args.sidecar else ""))
    print("re-detect before publishing: inkmark detect " + str(out) + " --key " + str(args.key))
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    keys = [_load_key(Path(p)) for p in args.key]
    table = default_table(args.tier)
    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8")) if args.sidecar else None

    if len(keys) == 1:
        decision = scan(text, keys[0], table=table, sidecar=sidecar)
    else:
        decision = scan_against(text, keys, table=table, sidecar=sidecar)

    print(decision.render())

    if args.payload_bits:
        best = max(keys, key=lambda k: scan(text, k, table=table).z)
        read = read_payload(text, best, args.payload_bits, table=table)
        value = read.payload
        if value is None:
            print(
                f"payload      undecoded: {read.unresolved} of {args.payload_bits} bits tied, "
                f"{read.inconclusive} slots carried no information"
            )
            print("             this is the tier-2 discovery path failing the way it should: "
                  "the text is too short for the bits, not evidence against the writer.")
        else:
            writer_id, published, crc_ok = decode_identity(value, args.payload_bits)
            print(f"payload      0x{value:x} -> writer id 0x{writer_id:x}, published {published}")
            print(f"checksum     {'ok' if crc_ok else 'MISMATCH - decode is not trustworthy'}")
            print(f"weakest bit  p = {read.weakest:.3g}")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    table = default_table(args.tier)
    print(f"tier {args.tier}: {table.class_count} classes, {table.member_count} surfaces")
    problems = table.audit()
    if problems:
        print("surfaces claimed by more than one class (both guards must disambiguate):")
        for line in problems:
            print(f"  {line}")
    if args.slots_from:
        text = Path(args.slots_from).read_text(encoding="utf-8")
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

    text = Path(args.file).read_text(encoding="utf-8")
    key = _load_key(Path(args.key))
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
    out = Path(args.out or f"{args.file}.inkmark.json")
    out.write_text(manifest.to_json(), encoding="utf-8")
    print(f"manifest     {out}")
    print(f"text sha256  {manifest.digest[:32]}...  (over the canonical form, not the file)")
    print(f"scheme       {manifest.scheme}")
    if args.domain:
        print(f"scoped to    {args.domain} -- hand that subkey to this publisher only")
    if not ed25519_available():
        print(
            "\n  ! No public signature: the attestation key is also the signing key, so you\n"
            "    must nominate a verifier rather than let the world check. Install\n"
            "    'inkmark[sign]' for Ed25519. Not a C2PA manifest and not a substitute for one."
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .manifest import Manifest, verify

    text = Path(args.file).read_text(encoding="utf-8")
    manifest = Manifest.from_json(Path(args.manifest).read_text(encoding="utf-8"))
    key = _load_key(Path(args.key))
    check = verify(canonical_form(text), manifest, key.for_domain(args.domain).secret if args.domain else key.secret)
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
    path = Path(args.file)
    registry = Registry.from_json(path.read_text(encoding="utf-8")) if path.exists() else Registry()
    if args.add:
        key = _load_key(Path(args.key))
        entry = entry_for(key, args.add, args.url or "")
        registry.add(entry)
        path.write_text(registry.to_json(), encoding="utf-8")
        print(f"added {entry.writer_id} -> {entry.handle} ({path})")
        print("Host this file yourself, ideally at /.well-known/inkmark.json")
    else:
        print(registry.to_json())
        print(f"\n{len(registry)} entries")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import math

    text = Path(args.file).read_text(encoding="utf-8")
    key = _load_key(Path(args.key))
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
    parser = argparse.ArgumentParser(prog="inkmark", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"inkmark {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    key = sub.add_parser("key", help="manage your writer key")
    key_sub = key.add_subparsers(dest="keycmd", required=True)
    new = key_sub.add_parser("new", help="create a keyring")
    new.add_argument("--out", default="~/.inkmark/keyring.json")
    new.add_argument("--note", default="")
    new.set_defaults(func=cmd_key_new)
    show = key_sub.add_parser("show", help="print the public fingerprint")
    show.add_argument("--key", default="~/.inkmark/keyring.json")
    show.set_defaults(func=cmd_key_show)

    em = sub.add_parser("embed", help="mark prose you wrote")
    em.add_argument("file")
    em.add_argument("--key", default="~/.inkmark/keyring.json")
    em.add_argument("--tier", default="ab", help="a, ab, or all (locale+register classes)")
    em.add_argument("--payload", metavar="HANDLE,YYYY-MM-DD", help="tier-2: encode identity+date")
    em.add_argument("--out", default=None, help="omit to preview only")
    em.add_argument("--sidecar", default=None, help="also write the slot ledger")
    em.add_argument("--interactive", action="store_true", help="veto each edit")
    em.add_argument(
        "--anti-slop",
        action="store_true",
        help="keep your own word where the key would push it toward machine-sounding filler. "
        "Costs statistical strength; `inkmark audit` prices it.",
    )
    em.set_defaults(func=cmd_embed)

    de = sub.add_parser("detect", help="test a suspect copy against a key")
    de.add_argument("file")
    de.add_argument("--key", action="append", required=True, help="repeatable for registry search")
    de.add_argument("--tier", default="ab")
    de.add_argument("--sidecar", default=None)
    de.add_argument("--payload-bits", type=int, default=0)
    de.set_defaults(func=cmd_detect)

    ta = sub.add_parser("table", help="inspect the equivalence table")
    ta.add_argument("--tier", default="ab")
    ta.add_argument("--slots-from", default=None, metavar="FILE")
    ta.set_defaults(func=cmd_table)

    pub = sub.add_parser("publish", help="write a signed provenance manifest next to the in-band mark")
    pub.add_argument("file")
    pub.add_argument("--key", default="~/.inkmark/keyring.json")
    pub.add_argument("--handle", required=True)
    pub.add_argument("--published", default=str(date.today()))
    pub.add_argument("--url", default="")
    pub.add_argument("--license", default="")
    pub.add_argument("--domain", default=None, help="scope the attestation to one publisher")
    pub.add_argument("--out", default=None)
    pub.set_defaults(func=cmd_publish)

    ver = sub.add_parser("verify", help="check a copy against a manifest")
    ver.add_argument("file")
    ver.add_argument("--manifest", required=True)
    ver.add_argument("--key", default="~/.inkmark/keyring.json")
    ver.add_argument("--domain", default=None)
    ver.set_defaults(func=cmd_verify)

    reg = sub.add_parser("registry", help="the self-hosted id -> handle file for tier 2")
    reg.add_argument("--file", default="inkmark-registry.json")
    reg.add_argument("--add", default=None, metavar="HANDLE")
    reg.add_argument("--url", default="")
    reg.add_argument("--key", default="~/.inkmark/keyring.json")
    reg.set_defaults(func=cmd_registry)

    au = sub.add_parser("audit", help="what this article can support before you publish it")
    au.add_argument("file")
    au.add_argument("--key", default="~/.inkmark/keyring.json")
    au.add_argument("--tier", default="ab")
    au.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for field in ("key", "out", "sidecar"):
        value = getattr(args, field, None)
        if isinstance(value, str) and value.startswith("~"):
            setattr(args, field, str(Path(value).expanduser()))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
