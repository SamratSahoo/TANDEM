"""Settings, credentials (write-only) and runtime status."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from tandem import __version__
from tandem.core import paths, probe, secrets
from tandem.core import runtime as runtime_mod
from tandem.core import settings as settings_mod

router = APIRouter(tags=["settings"])


class SettingsBody(BaseModel):
    data_root: str | None = None
    hf_org: str | None = None
    ui_port: int | None = None


class SecretBody(BaseModel):
    """Write-only. A secret never travels back out of this API."""

    gemini_api_key: str | None = None
    hf_token: str | None = None


@router.get("/settings")
async def get_settings() -> dict:
    cfg = settings_mod.load(force=True)
    return {
        "version": __version__,
        "settings": {key: value for key, value in settings_mod.flatten(cfg)},
        "paths": {
            "config": str(paths.config_file()),
            "credentials": str(paths.credentials_file()),
            "data_root": str(cfg.resolved_data_root()),
            "runtime": str(cfg.resolved_runtime_dir()),
            "logs": str(paths.log_dir()),
        },
        "credentials": {
            # Source and mask only — enough for the UI to say which key is in play, never
            # enough to use one.
            "gemini": {"source": secrets.gemini_key_source(), "masked": secrets.mask(secrets.gemini_api_key())},
            "hf": {"source": secrets.hf_token_source(), "masked": secrets.mask(secrets.hf_token())},
        },
    }


@router.put("/settings")
async def update_settings(body: SettingsBody) -> dict:
    cfg = settings_mod.load()
    if body.data_root is not None:
        cfg.data_root = body.data_root
    if body.hf_org is not None:
        cfg.hf_org = body.hf_org
    if body.ui_port is not None:
        cfg.ui.port = body.ui_port
    settings_mod.save(cfg)
    return await get_settings()


@router.put("/settings/secrets")
async def update_secrets(body: SecretBody) -> dict:
    if body.gemini_api_key:
        secrets.set_gemini_api_key(body.gemini_api_key)
    if body.hf_token:
        secrets.set_hf_token(body.hf_token)
    return await get_settings()


@router.get("/runtime")
async def runtime_status() -> dict:
    cfg = settings_mod.load()
    runtime = runtime_mod.Runtime(cfg.resolved_runtime_dir())
    status = runtime.status()
    return {**status.to_dict(), "root": str(runtime.root)}


@router.get("/doctor")
async def doctor(profile: str | None = None, hardware: bool = False) -> dict:
    from tandem.cli.doctor import collect_checks

    checks = collect_checks(profile_name=profile, probe_hardware=hardware)
    summary = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    for check in checks:
        summary[check.state] = summary.get(check.state, 0) + 1
    return {"checks": [c.to_dict() for c in checks], "summary": summary}


@router.get("/probe/gemini")
async def probe_gemini() -> dict:
    return probe.check_gemini_key().to_dict()
