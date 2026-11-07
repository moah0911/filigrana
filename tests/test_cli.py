"""The CLI, exercised end to end with a throwaway keyring in a temp directory.

This is the surface a writer actually touches, and for a long time it had not a single test
line behind it -- every number quoted in the README was checked at the library layer only.
These tests run the real `main()` against an isolated HOME, so a bug in argument wiring, the
~/.inkmark default, or the rendered output shows up here rather than in a writer's hands.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import KEY as KEY_A

from inkmark.cli import main

PASSPHRASE = "correct horse"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HOME + INKMARK_PASSPHRASE so nothing real is touched."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("INKMARK_PASSPHRASE", PASSPHRASE)
    monkeypatch.chdir(tmp_path)
    keyfile = tmp_path / ".inkmark" / "keyring.json"
    KEY_A.save(keyfile, PASSPHRASE)
    return tmp_path


def run(*argv: str) -> int:
    return main(list(argv))


def _key_path(home: Path) -> Path:
    return home / ".inkmark" / "keyring.json"


class TestKeyCommands:
    def test_key_new_writes_an_encrypted_ring_and_prints_the_fingerprint(self, home, capsys):
        assert run("key", "new", "--out", str(_key_path(home))) == 0
        out = capsys.readouterr().out
        assert home / ".inkmark" / "keyring.json" in list(home.rglob("keyring.json"))
        blob = json.loads((_key_path(home)).read_text())
        assert blob["format"] == "inkmark-keyring/1"
        assert blob["fingerprint"] in out
        mode = (_key_path(home)).stat().st_mode & 0o777
        assert mode == 0o600, "must refuse group/world access, or anyone can forge the writer"

    def test_key_show_reports_key_id_and_writer_id(self, home, capsys):
        from inkmark.codec import writer_id_from

        assert run("key", "show", "--key", str(_key_path(home))) == 0
        out = capsys.readouterr().out
        assert "gate-test" in out
        expected = f"0x{writer_id_from(KEY_A.fingerprint):x}"
        assert expected in out

    def test_detect_refuses_a_missing_keyring_with_a_usable_hint(self, home, capsys):
        missing = home / "nowhere" / "keyring.json"
        essay = home / "essay.md"
        essay.write_text("she walked slowly toward the grey station")
        with pytest.raises(SystemExit) as exc:
            run("detect", str(essay), "--key", str(missing), "--key", str(_key_path(home)))
        assert "inkmark key new" in str(exc.value)


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
        assert record["format"] == "inkmark-sidecar/2"
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
        from inkmark.embed import embed

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
        from inkmark.keyring import WriterKey

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
        manifest = home / "essay.inkmark.json"
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
        manifest = home / "essay.inkmark.json"
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
        from inkmark.codec import writer_id_from

        registry = home / "reg.json"
        assert run("registry", "--file", str(registry), "--add", "ada", "--key", str(_key_path(home))) == 0
        record = json.loads(registry.read_text())
        expected = f"{writer_id_from(KEY_A.fingerprint):06x}"
        assert record["entries"][0]["writer_id"] == expected
        capsys.readouterr()
        assert run("registry", "--file", str(registry)) == 0
        assert expected in capsys.readouterr().out


class TestPassphraseHandling:
    def test_env_passphrase_beats_the_prompt_and_bad_passphrase_is_named(self, home, monkeypatch):
        monkeypatch.setenv("INKMARK_PASSPHRASE", "wrong")
        with pytest.raises(SystemExit) as exc:
            run("key", "show")
        assert "bad passphrase" in str(exc.value)
        monkeypatch.delenv("INKMARK_PASSPHRASE")
        monkeypatch.setattr("builtins.input", lambda *a, _k="": "")
        monkeypatch.setattr(sys, "stdin", open(os.devnull))

    def test_mismatched_confirmation_never_mints_a_keyring(self, home, monkeypatch):
        rings_before = list(home.rglob("keyring.json"))
        monkeypatch.delenv("INKMARK_PASSPHRASE", raising=False)
        import getpass

        answers = iter(["one", "two"])
        monkeypatch.setattr(getpass, "getpass", lambda *a: next(answers))
        with pytest.raises(SystemExit) as exc:
            run("key", "new", "--out", str(home / "fresh.json"))
        assert "did not match" in str(exc.value)
        assert list(home.rglob("keyring.json")) == rings_before
