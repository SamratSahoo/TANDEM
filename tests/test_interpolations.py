"""OmegaConf interpolations in what tandem reads, and URLs that cannot be used.

The source monorepo's configs are OmegaConf, so values written from them carry ``${oc.env:VAR,default}``
interpolations that only mean something inside that framework. A profile has to be concrete: anything
left unresolved is a string that looks like configuration and behaves like a crash. (These were the
importer's tests; the importer is gone, and reading a profile still resolves them.)
"""

from __future__ import annotations

import pytest

from tandem.core import probe
from tandem.core import rig as rig_mod
from tandem.core.errors import RigInvalid
from tandem.core.profiles import resolve_interpolation
from tandem.planners.tiptop import probe as tiptop_probe
from tandem.planners.tiptop.options import resolve_profile


class TestResolveInterpolation:
    """OmegaConf interpolations must not survive into a profile."""

    def test_whole_string_interpolation(self, monkeypatch):
        monkeypatch.delenv("TIPTOP_HAND_CAMERA_ID", raising=False)
        assert resolve_interpolation("${oc.env:TIPTOP_HAND_CAMERA_ID,14846828}") == "14846828"

    def test_embedded_interpolation(self, monkeypatch):
        """The one that bit: the port sits INSIDE a URL, so an anchored match missed it and
        the raw `${...}` was written to the profile, where urlparse later exploded on it."""
        monkeypatch.delenv("TIPTOP_M2T2_PORT", raising=False)
        assert resolve_interpolation("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}") == "http://localhost:8123"

    def test_environment_wins_over_the_default(self, monkeypatch):
        monkeypatch.setenv("TIPTOP_M2T2_PORT", "9001")
        assert resolve_interpolation("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}") == "http://localhost:9001"

    def test_interpolation_without_a_default(self, monkeypatch):
        monkeypatch.setenv("SOME_VAR", "resolved")
        assert resolve_interpolation("${oc.env:SOME_VAR}") == "resolved"

    def test_unresolvable_interpolation_is_left_alone(self, monkeypatch):
        """Nothing to resolve it to. Left intact so validation can name it, rather than
        silently becoming an empty string that fails somewhere else."""
        monkeypatch.delenv("NEVER_SET_VAR", raising=False)
        assert resolve_interpolation("${oc.env:NEVER_SET_VAR}") == "${oc.env:NEVER_SET_VAR}"

    def test_several_in_one_string(self, monkeypatch):
        monkeypatch.delenv("A_HOST", raising=False)
        monkeypatch.delenv("A_PORT", raising=False)
        assert resolve_interpolation("http://${oc.env:A_HOST,box}:${oc.env:A_PORT,80}/x") == "http://box:80/x"

    def test_plain_values_pass_through(self):
        assert resolve_interpolation("http://localhost:8123") == "http://localhost:8123"
        assert resolve_interpolation(15) == 15
        assert resolve_interpolation(None) is None


class TestUrlValidation:
    """A URL that cannot be parsed is caught where it is set -- rig.yml, now -- not three layers down."""

    def _rig(self, url: str) -> str:
        return f"version: 1\nplanners:\n  tiptop:\n    perception:\n      m2t2: {{url: '{url}'}}\n"

    def test_an_unresolved_interpolation_is_rejected(self):
        with pytest.raises(RigInvalid) as excinfo:
            rig_mod.parse_text(self._rig("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}"), source="rig.yml")
        message = excinfo.value.message
        assert "planners.tiptop.perception.m2t2.url" in message
        assert "8123" in message, "the message should show the value that failed"

    def test_a_url_without_a_host_is_rejected(self):
        with pytest.raises(RigInvalid, match="m2t2"):
            rig_mod.parse_text(self._rig("not-a-url"), source="rig.yml")

    def test_a_good_url_passes(self, machine_rig):
        rig = rig_mod.update({"planners.tiptop.perception.m2t2.url": "http://10.0.0.4:8123"})
        from tandem.core.profiles import Profile

        assert resolve_profile(Profile(name="x"), rig).perception.m2t2.url == "http://10.0.0.4:8123"


class TestProbeRobustness:
    """`doctor` is what you run WHEN something is wrong, so no probe may crash the run."""

    @pytest.mark.parametrize("check", [tiptop_probe.check_m2t2, tiptop_probe.check_foundation_stereo])
    def test_an_unparseable_url_is_a_failed_check_not_an_exception(self, check):
        row = check("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}")
        assert row.state == probe.FAIL
        assert row.hint and "tandem rig set planners.tiptop.perception." in row.hint
        assert "${" in row.detail

    @pytest.mark.parametrize("check", [tiptop_probe.check_m2t2, tiptop_probe.check_foundation_stereo])
    def test_a_nonsense_url_is_a_failed_check(self, check):
        assert check("").state == probe.FAIL
        assert check("://////").state == probe.FAIL

    def test_a_reachable_looking_url_still_probes(self):
        # Nothing is listening, so this warns rather than fails — the distinction being that
        # the URL is usable and the server merely is not up yet.
        assert tiptop_probe.check_m2t2("http://127.0.0.1:1").state == probe.WARN
        row = tiptop_probe.check_foundation_stereo("http://127.0.0.1:1")
        assert row.state == probe.WARN
        assert row.name == "foundation stereo depth server"


class TestStoredProfilesSelfHeal:
    """A profile file that still holds an interpolation, written from one of the monorepo's configs."""

    def test_a_stored_interpolation_still_loads(self, profile, monkeypatch):
        """Turning a probe crash into a profile that cannot be opened at all would be a
        worse outcome, so reading resolves too — the file heals on the next save."""
        from tandem.core import profiles

        monkeypatch.delenv("TANDEM_TEST_PROPOSER", raising=False)
        text = profile.file().read_text().replace(
            "proposal_model: gemini-2.5-pro", "proposal_model: ${oc.env:TANDEM_TEST_PROPOSER,gemini-2.5-pro}"
        )
        assert "${oc.env" in text
        profile.file().write_text(text)

        assert profiles.load(profile.name).hitl.proposal_model == "gemini-2.5-pro"

    def test_saving_writes_the_resolved_value_back(self, profile, monkeypatch):
        from tandem.core import profiles

        monkeypatch.delenv("TANDEM_TEST_PROPOSER", raising=False)
        text = profile.file().read_text().replace(
            "proposal_model: gemini-2.5-pro", "proposal_model: ${oc.env:TANDEM_TEST_PROPOSER,gemini-2.5-pro}"
        )
        profile.file().write_text(text)

        profiles.save(profiles.load(profile.name))
        assert "${oc.env" not in profile.file().read_text()
