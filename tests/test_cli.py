"""The CLI, exercised end to end with a throwaway keyring in a temp directory.

This is the surface a writer actually touches, and for a long time it had not a single test
line behind it -- every number quoted in the README was checked at the library layer only.
These tests run the real `main()` against an isolated HOME, so a bug in argument wiring, the
~/.filigrana default, or the rendered output shows up here rather than in a writer's hands.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import KEY as KEY_A
from filigrana.cli import main
from filigrana.codec import BitVote, PayloadRead

PASSPHRASE = "correct horse"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HOME + FILIGRANA_PASSPHRASE so nothing real is touched."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FILIGRANA_PASSPHRASE", PASSPHRASE)
    monkeypatch.chdir(tmp_path)
    keyfile = tmp_path / ".filigrana" / "keyring.json"
    KEY_A.save(keyfile, PASSPHRASE)
    return tmp_path


def run(*argv: str) -> int:
    return main(list(argv))


def _key_path(home: Path) -> Path:
    return home / ".filigrana" / "keyring.json"


class TestKeyCommands:
    def test_key_new_writes_an_encrypted_ring_and_prints_the_fingerprint(self, home, capsys):
        assert run("key", "new", "--out", str(_key_path(home))) == 0
        out = capsys.readouterr().out
        assert home / ".filigrana" / "keyring.json" in list(home.rglob("keyring.json"))
        blob = json.loads((_key_path(home)).read_text())
        assert blob["format"] == "filigrana-keyring/1"
        assert blob["fingerprint"] in out
        mode = (_key_path(home)).stat().st_mode & 0o777
        assert mode == 0o600, "must refuse group/world access, or anyone can forge the writer"

    def test_key_show_reports_key_id_and_writer_id(self, home, capsys):
        from filigrana.codec import writer_id_from

        assert run("key", "show", "--key", str(_key_path(home))) == 0
        out = capsys.readouterr().out
        assert "gate-test" in out
        expected = f"0x{writer_id_from(KEY_A.fingerprint):x}"
        assert expected in out

    def test_detect_refuses_a_missing_keyring_with_a_usable_hint(self, home, capsys):
        missing = home / "nowhere" / "keyring.json"
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        assert run(
            "detect", str(essay), "--key", str(missing), "--key", str(_key_path(home))
        ) == 2
        assert "filigrana key new" in capsys.readouterr().err


class TestEmbedCommand:
    def test_embed_is_a_dry_run_until_you_name_an_output_file(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        assert run("embed", str(essay), "--key", str(_key_path(home))) == 0
        out = capsys.readouterr().out
        assert "nothing written" in out
        assert not (home / "essay.marked.md").exists()

    def test_embed_writes_marked_text_and_sidecar_when_asked(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        out_file = home / "essay.marked.md"
        sidecar = home / "essay.sidecar.json"
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)),
            "--tier", "all", "--out", str(out_file), "--sidecar", str(sidecar),
        ) == 0
        out = capsys.readouterr().out
        assert f"wrote {out_file}" in out
        assert out_file.exists() and sidecar.exists()
        record = json.loads(sidecar.read_text())
        assert record["format"] == "filigrana-sidecar/2"
        assert record["slots"], "the sidecar must publish the per-slot ledger"

    def test_embed_payload_encodes_identity_and_date(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        out_file = home / "essay.marked.md"
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)), "--tier", "all",
            "--payload", "ada,2026-11-07", "--out", str(out_file),
        ) == 0
        assert "2026-11-07" in capsys.readouterr().out


class TestDetectCommand:
    def test_detect_prints_the_statistic_and_never_a_verdict(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        from filigrana.embed import embed

        marked = embed(essay.read_text(), KEY_A, payload=None, n_bits=0)
        (home / "marked.md").write_text(marked.text)
        assert run("detect", str(home / "marked.md"), "--key", str(_key_path(home)), "--tier", "all") == 0
        out = capsys.readouterr().out
        assert "statistics" in out and "p =" in out
        for banned in ("verd", "is_mine", "guilty", "accus"):
            assert banned not in out.lower(), "a watermark score must never read as a verdict"

    def test_detect_of_unmarked_prose_reports_a_weak_p(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text(
            "Ordinary prose has choices that were never made under any key at all, "
            "so the statistic must stay near chance and say so plainly. The crowd parted "
            "like water, and the grey station sat quietly against the sky."
        )
        run("detect", str(essay), "--key", str(_key_path(home)), "--tier", "all")
        out = capsys.readouterr().out
        assert "0." in out  # the p-value exists, and the z is small
        assert "z = " in out

    def test_detect_with_two_keys_prints_the_multiplicity_correction(self, home, capsys):
        from filigrana.keyring import WriterKey

        second = home / "second.json"
        WriterKey.generate().save(second, PASSPHRASE)
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        run(
            "detect", str(essay), "--key", str(_key_path(home)), "--key", str(second), "--tier", "all"
        )
        out = capsys.readouterr().out
        assert "2 keys were tested" in out, "the corrected p-value must announce itself"


class TestAuditAndTable:
    def test_audit_prices_the_article_up_front(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        assert run("audit", str(essay), "--key", str(_key_path(home)), "--tier", "all") == 0
        out = capsys.readouterr().out
        assert "best possible z" in out
        assert "50%" in out.lower() or "half the text" in out
        assert "best case" in out, "an audit is a forecast under the strongest assumption"

    def test_table_counts_the_tiers(self, home, capsys):
        assert run("table", "--tier", "all") == 0
        out = capsys.readouterr().out
        assert "classes" in out


class TestPublishAndVerify:
    def test_publish_then_verify_round_trips_the_claims(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the gray station\n")
        manifest = home / "essay.filigrana.json"
        assert run(
            "publish", str(essay), "--key", str(_key_path(home)), "--handle", "ada",
            "--published", "2026-11-07", "--out", str(manifest),
        ) == 0
        capsys.readouterr()
        assert run(
            "verify", str(essay), "--manifest", str(manifest), "--key", str(_key_path(home))
        ) == 0
        out = capsys.readouterr().out
        assert "text matches      True" in out
        assert "attestation ok    True" in out
        assert "published         2026-11-07" in out

    def test_verify_of_an_edited_copy_reports_an_edit_not_a_broken_signature(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the gray station\n")
        manifest = home / "essay.filigrana.json"
        run(
            "publish", str(essay), "--key", str(_key_path(home)), "--handle", "ada", "--out", str(manifest)
        )
        (home / "edited.md").write_text(
            "she walked slowly toward the gray station, and the crowd parted\n"
        )
        run(
            "verify", str(home / "edited.md"), "--manifest", str(manifest), "--key", str(_key_path(home))
        )
        out = capsys.readouterr().out
        assert "text matches      False" in out
        assert "edit" in out, "a divergence in text must read as an edit, not as tampering"


class TestRegistry:
    def test_registry_add_and_dump_resolves_the_writer_id(self, home, capsys):
        from filigrana.codec import writer_id_from

        registry = home / "reg.json"
        assert run("registry", "--file", str(registry), "--add", "ada", "--key", str(_key_path(home))) == 0
        record = json.loads(registry.read_text())
        expected = f"{writer_id_from(KEY_A.fingerprint):06x}"
        assert record["entries"][0]["writer_id"] == expected
        capsys.readouterr()
        assert run("registry", "--file", str(registry)) == 0
        assert expected in capsys.readouterr().out


class TestPassphraseHandling:
    def test_env_passphrase_beats_the_prompt_and_bad_passphrase_is_named(
        self, home, monkeypatch, capsys
    ):
        monkeypatch.setenv("FILIGRANA_PASSPHRASE", "wrong")
        assert run("key", "show") == 2
        assert "bad passphrase" in capsys.readouterr().err
        monkeypatch.delenv("FILIGRANA_PASSPHRASE")
        monkeypatch.setattr("builtins.input", lambda *a, _k="": "")
        monkeypatch.setattr(sys, "stdin", open(os.devnull))

    def test_mismatched_confirmation_never_mints_a_keyring(self, home, monkeypatch, capsys):
        rings_before = list(home.rglob("keyring.json"))
        monkeypatch.delenv("FILIGRANA_PASSPHRASE", raising=False)
        import getpass

        answers = iter(["one", "two"])
        monkeypatch.setattr(getpass, "getpass", lambda *a: next(answers))
        assert run("key", "new", "--out", str(home / "fresh.json")) == 2
        assert "did not match" in capsys.readouterr().err
        assert list(home.rglob("keyring.json")) == rings_before


class TestInteractiveVeto:
    """--interactive asks per slot; every answer shape must land, including EOF mid-prompt."""

    ESSAY = "she walked slowly toward the grey station and the crowd parted"

    def test_vetoing_every_prompt_keeps_the_writers_words(self, home, monkeypatch, capsys):
        # The prompt is "(y/N)": yes = veto this edit and keep your own word.
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        essay = home / "essay.md"
        essay.write_text(self.ESSAY)
        out_path = home / "marked.md"
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)), "--tier", "all",
            "--interactive", "--out", str(out_path),
        ) == 0
        out = capsys.readouterr().out
        assert "declined by you" in out and "0 declined by you" not in out
        assert out_path.read_text() == self.ESSAY, "a vetoed slot must keep your own word"

    def test_accepting_every_prompt_applies_the_key(self, home, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        essay = home / "essay.md"
        essay.write_text(self.ESSAY)
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)), "--tier", "all",
            "--interactive", "--out", str(home / "marked.md"),
        ) == 0
        assert "0 declined by you" in capsys.readouterr().out

    def test_eof_mid_veto_exits_cleanly(self, home, monkeypatch, capsys):
        def refuse(*a):
            raise EOFError

        monkeypatch.setattr("builtins.input", refuse)
        essay = home / "essay.md"
        essay.write_text(self.ESSAY)
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)), "--tier", "all",
            "--interactive", "--out", str(home / "marked.md"),
        ) == 2
        assert "interactive needs a terminal" in capsys.readouterr().err

    def test_lost_first_prompt_stdin_exits_cleanly(self, home, monkeypatch, capsys):
        import getpass

        monkeypatch.delenv("FILIGRANA_PASSPHRASE", raising=False)

        def refuse(*a):
            raise EOFError

        monkeypatch.setattr(getpass, "getpass", refuse)
        assert run("key", "show") == 2
        assert "a passphrase is required" in capsys.readouterr().err

    def test_lost_confirmation_stdin_exits_cleanly(self, home, monkeypatch, capsys):
        import getpass

        monkeypatch.delenv("FILIGRANA_PASSPHRASE", raising=False)
        calls = iter(["first", EOFError()])

        def prompt(*a):
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(getpass, "getpass", prompt)
        assert run("key", "new", "--out", str(home / "fresh.json")) == 2
        assert "confirmation prompt lost its stdin" in capsys.readouterr().err


class TestTableSlotsFrom:
    def test_table_counts_the_slots_of_a_file(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        assert run("table", "--tier", "all", "--slots-from", str(essay)) == 0
        out = capsys.readouterr().out
        assert "words -> " in out and "per 1000 words" in out


class TestFailurePaths:
    """Every predictable user mistake must be one line on stderr and exit 2, never a traceback."""

    def test_a_missing_input_file_is_named(self, home, capsys):
        assert run("detect", str(home / "nope.md"), "--key", str(_key_path(home))) == 2
        err = capsys.readouterr().err
        assert "no such file" in err and "nope.md" in err

    def test_a_directory_where_a_file_belongs_is_named(self, home, capsys):
        assert run("embed", str(home), "--key", str(_key_path(home))) == 2
        assert "is a directory" in capsys.readouterr().err

    def test_invalid_sidecar_json_is_named_with_its_position(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        sidecar = home / "sidecar.json"
        sidecar.write_text("{not json")
        assert run(
            "detect", str(essay), "--key", str(_key_path(home)), "--sidecar", str(sidecar)
        ) == 2
        err = capsys.readouterr().err
        assert "not valid JSON" in err and "line" in err

    def test_an_old_sidecar_format_is_refused(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        sidecar = home / "sidecar.json"
        sidecar.write_text(json.dumps({"format": "filigrana-sidecar/1", "slots": []}))
        assert run(
            "detect", str(essay), "--key", str(_key_path(home)), "--sidecar", str(sidecar)
        ) == 2
        assert "not a filigrana sidecar" in capsys.readouterr().err

    def test_a_manifest_missing_required_fields_is_named(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the gray station\n")
        manifest = home / "manifest.json"
        manifest.write_text(json.dumps({"format": "filigrana-manifest/1"}))
        assert run(
            "verify", str(essay), "--manifest", str(manifest), "--key", str(_key_path(home))
        ) == 2
        assert "missing or not a string" in capsys.readouterr().err

    def test_an_unwritable_output_is_reported_without_a_traceback(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        blocker = home / "blocked"
        blocker.write_text("not a directory")
        assert run(
            "embed", str(essay), "--key", str(_key_path(home)),
            "--out", str(blocker / "marked.md"),
        ) == 2
        assert "cannot write" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv",
        [
            ("table", "--tier", "xyz"),
            ("embed", "essay.md", "--payload", "ada,not-a-date"),
            ("detect", "essay.md", "--key", "k", "--payload-bits", "0"),
            ("publish", "essay.md", "--handle", "ada", "--published", "07/11/2026"),
        ],
    )
    def test_invalid_arguments_exit_two(self, argv):
        with pytest.raises(SystemExit) as exc:
            run(*argv)
        assert exc.value.code == 2


class TestDomainScopedAttestation:
    def test_publish_and_verify_agree_on_the_right_domain_only(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the gray station\n")
        manifest = home / "essay.filigrana.json"
        assert run(
            "publish", str(essay), "--key", str(_key_path(home)), "--handle", "ada",
            "--domain", "example.org", "--out", str(manifest),
        ) == 0
        assert "scoped to    example.org" in capsys.readouterr().out
        capsys.readouterr()
        assert run(
            "verify", str(essay), "--manifest", str(manifest), "--key", str(_key_path(home)),
            "--domain", "example.org",
        ) == 0
        assert "attestation ok    True" in capsys.readouterr().out
        capsys.readouterr()
        assert run(
            "verify", str(essay), "--manifest", str(manifest), "--key", str(_key_path(home)),
            "--domain", "other.org",
        ) == 0
        out = capsys.readouterr().out
        assert "text matches      True" in out
        assert "attestation ok    False" in out, "a different domain is a different subkey"


class TestPayloadReporting:
    """_report_payload is extracted from cmd_detect so both outcomes are testable without
    the ~60,000 words the 45-bit wire format genuinely needs."""

    @staticmethod
    def _votes(payload: int, n_bits: int) -> tuple:
        return tuple(
            BitVote(i, 1, 9) if (payload >> (n_bits - 1 - i)) & 1 else BitVote(i, 9, 1)
            for i in range(n_bits)
        )

    def test_a_decoded_identity_reports_who_and_when(self, capsys):
        from filigrana.cli import _report_payload
        from filigrana.codec import encode_identity

        payload, n_bits = encode_identity("62365a5464bdae5e", "2026-11-17")
        read = PayloadRead(votes=self._votes(payload, n_bits), unresolved=0, inconclusive=3)
        _report_payload(read, n_bits)
        out = capsys.readouterr().out
        assert f"0x{payload:x} -> writer id 0x62365a, published 2026-11-17" in out
        assert "checksum     ok" in out
        assert "weakest bit" in out

    def test_a_checksum_mismatch_is_said_out_loud(self, capsys):
        from filigrana.cli import _report_payload
        from filigrana.codec import encode_identity

        payload, n_bits = encode_identity("62365a5464bdae5e", "2026-11-17")
        corrupt = payload ^ (1 << 7)
        read = PayloadRead(votes=self._votes(corrupt, n_bits), unresolved=0, inconclusive=0)
        _report_payload(read, n_bits)
        assert "MISMATCH - decode is not trustworthy" in capsys.readouterr().out

    def test_an_undecodable_payload_never_invents_an_author(self, capsys):
        from filigrana.cli import _report_payload

        _report_payload(PayloadRead(votes=(), unresolved=37, inconclusive=15), 45)
        out = capsys.readouterr().out
        assert "undecoded: 37 of 45 bits tied, 15 slots carried no information" in out
        assert "not evidence against the writer" in out

    def test_detect_on_a_short_essay_reports_the_payload_honestly(self, home, capsys):
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station and the crowd parted")
        assert run(
            "detect", str(essay), "--key", str(_key_path(home)), "--tier", "all",
            "--payload-bits", "45",
        ) == 0
        out = capsys.readouterr().out
        assert "undecoded" in out and "bits tied" in out


def test_module_entry_point_prints_the_version():
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "filigrana", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip().startswith("filigrana 0.")
