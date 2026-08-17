"""Importing an existing rig's setup into a profile.

The source configs are OmegaConf, so values arrive carrying ``${oc.env:VAR,default}``
interpolations that only mean something inside that framework. A profile has to be concrete:
anything left unresolved is a string that looks like configuration and behaves like a crash.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tandem.core import probe
from tandem.core.importers import _deref
from tandem.core.profiles import Profile


class TestDeref:
    """OmegaConf interpolations must not survive into a profile."""

    def test_whole_string_interpolation(self, monkeypatch):
        monkeypatch.delenv("TIPTOP_HAND_CAMERA_ID", raising=False)
        assert _deref("${oc.env:TIPTOP_HAND_CAMERA_ID,14846828}") == "14846828"

    def test_embedded_interpolation(self, monkeypatch):
        """The one that bit: the port sits INSIDE a URL, so an anchored match missed it and
        the raw `${...}` was written to the profile, where urlparse later exploded on it."""
        monkeypatch.delenv("TIPTOP_M2T2_PORT", raising=False)
        assert _deref("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}") == "http://localhost:8123"

    def test_environment_wins_over_the_default(self, monkeypatch):
        """Import captures what the rig actually resolves to, not what its file would say on
        a different machine."""
        monkeypatch.setenv("TIPTOP_M2T2_PORT", "9001")
        assert _deref("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}") == "http://localhost:9001"

    def test_interpolation_without_a_default(self, monkeypatch):
        monkeypatch.setenv("SOME_VAR", "resolved")
        assert _deref("${oc.env:SOME_VAR}") == "resolved"

    def test_unresolvable_interpolation_is_left_alone(self, monkeypatch):
        """Nothing to resolve it to. Left intact so validation can name it, rather than
        silently becoming an empty string that fails somewhere else."""
        monkeypatch.delenv("NEVER_SET_VAR", raising=False)
        assert _deref("${oc.env:NEVER_SET_VAR}") == "${oc.env:NEVER_SET_VAR}"

    def test_several_in_one_string(self, monkeypatch):
        monkeypatch.delenv("A_HOST", raising=False)
        monkeypatch.delenv("A_PORT", raising=False)
        assert _deref("http://${oc.env:A_HOST,box}:${oc.env:A_PORT,80}/x") == "http://box:80/x"

    def test_plain_values_pass_through(self):
        assert _deref("http://localhost:8123") == "http://localhost:8123"
        assert _deref(15) == 15
        assert _deref(None) is None


class TestUrlValidation:
    """A URL that cannot be parsed is caught where it is set, not three layers down."""

    def test_an_unresolved_interpolation_is_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            Profile.model_validate({
                "name": "x",
                "perception": {"m2t2": {"url": "http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}"}},
            })
        message = str(excinfo.value)
        assert "m2t2" in message
        assert "8123" in message, "the message should show the value that failed"

    def test_a_url_without_a_host_is_rejected(self):
        with pytest.raises(ValidationError):
            Profile.model_validate({"name": "x", "perception": {"m2t2": {"url": "not-a-url"}}})

    def test_a_good_url_passes(self):
        profile = Profile.model_validate({
            "name": "x", "perception": {"m2t2": {"url": "http://10.0.0.4:8123"}}
        })
        assert profile.perception.m2t2.url == "http://10.0.0.4:8123"


class TestProbeRobustness:
    """`doctor` is what you run WHEN something is wrong, so no probe may crash the run."""

    def test_an_unparseable_url_is_a_failed_check_not_an_exception(self):
        check = probe.check_m2t2("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}")
        assert check.state == probe.FAIL
        assert check.hint
        assert "${" in check.detail

    def test_a_nonsense_url_is_a_failed_check(self):
        assert probe.check_m2t2("").state == probe.FAIL
        assert probe.check_m2t2("://////").state == probe.FAIL

    def test_a_reachable_looking_url_still_probes(self):
        # Nothing is listening, so this warns rather than fails — the distinction being that
        # the URL is usable and the server merely is not up yet.
        check = probe.check_m2t2("http://127.0.0.1:1")
        assert check.state == probe.WARN


class TestStoredProfilesSelfHeal:
    """A profile written before the importer knew about embedded interpolations."""

    def test_a_stored_interpolation_still_loads(self, profile):
        """Turning a probe crash into a profile that cannot be opened at all would be a
        worse outcome, so reading resolves too — the file heals on the next save."""
        from tandem.core import profiles

        text = profile.profile_file().read_text().replace(
            "url: http://localhost:8123",
            "url: http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}",
        )
        profile.profile_file().write_text(text)

        loaded = profiles.load(profile.name)
        assert loaded.perception.m2t2.url == "http://localhost:8123"

    def test_saving_writes_the_resolved_value_back(self, profile):
        from tandem.core import profiles

        text = profile.profile_file().read_text().replace(
            "url: http://localhost:8123",
            "url: http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}",
        )
        profile.profile_file().write_text(text)

        profiles.save(profiles.load(profile.name))
        assert "${oc.env" not in profile.profile_file().read_text()
