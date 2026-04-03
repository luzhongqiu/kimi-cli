from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import UploadFile
from kaos.path import KaosPath

from kimi_cli.session import Session
from kimi_cli.web.api import sessions as sessions_api
from kimi_cli.web.runner import process as process_api
from kimi_cli.web.store.sessions import JointSession


@pytest.fixture
def isolated_share_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    share_dir = tmp_path / "share"
    share_dir.mkdir()

    def _get_share_dir() -> Path:
        share_dir.mkdir(parents=True, exist_ok=True)
        return share_dir

    monkeypatch.setattr("kimi_cli.share.get_share_dir", _get_share_dir)
    monkeypatch.setattr("kimi_cli.metadata.get_share_dir", _get_share_dir)
    return share_dir


@pytest.fixture
def work_dir(tmp_path: Path) -> KaosPath:
    path = tmp_path / "work"
    path.mkdir()
    return KaosPath.unsafe_from_local_path(path)


def _make_joint_session(kimi_session: Session) -> JointSession:
    return JointSession(
        session_id=UUID(kimi_session.id),
        title=kimi_session.title,
        last_updated=datetime.now(UTC),
        is_running=False,
        status=None,
        work_dir=str(kimi_session.work_dir),
        session_dir=str(kimi_session.dir),
        kimi_cli_session=kimi_session,
        archived=False,
    )


class _IdleRunner:
    def get_session(self, _session_id: UUID) -> None:
        return None


@pytest.mark.anyio
async def test_upload_session_file_saves_into_work_dir_uploads_and_deduplicates(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kimi_session = await Session.create(work_dir)
    joint_session = _make_joint_session(kimi_session)
    session_id = UUID(kimi_session.id)

    monkeypatch.setattr(
        sessions_api,
        "get_editable_session",
        lambda _session_id, _runner: joint_session,
    )

    first = UploadFile(filename="report.txt", file=BytesIO(b"first upload"))
    second = UploadFile(filename="report.txt", file=BytesIO(b"second upload"))

    first_result = await sessions_api.upload_session_file(
        session_id,
        first,
        runner=_IdleRunner(),
    )
    second_result = await sessions_api.upload_session_file(
        session_id,
        second,
        runner=_IdleRunner(),
    )

    uploads_dir = Path(str(work_dir)) / "uploads"
    assert (uploads_dir / "report.txt").read_text(encoding="utf-8") == "first upload"
    assert (uploads_dir / "report_1.txt").read_text(encoding="utf-8") == "second upload"
    assert first_result.filename == "report.txt"
    assert second_result.filename == "report_1.txt"
    assert first_result.path == str(uploads_dir / "report.txt")
    assert second_result.path == str(uploads_dir / "report_1.txt")
    assert not (kimi_session.dir / "uploads" / "report.txt").exists()


@pytest.mark.anyio
async def test_shared_work_dir_uploads_keep_sent_marker_session_scoped(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_a = await Session.create(work_dir)
    session_b = await Session.create(work_dir)
    joint_sessions = {
        UUID(session_a.id): _make_joint_session(session_a),
        UUID(session_b.id): _make_joint_session(session_b),
    }

    uploads_dir = Path(str(work_dir)) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    shared_file = uploads_dir / "shared.txt"
    shared_file.write_text("shared upload", encoding="utf-8")

    monkeypatch.setattr(
        process_api,
        "load_session_by_id",
        lambda session_id: joint_sessions[session_id],
    )
    monkeypatch.setattr(
        process_api,
        "load_config",
        lambda: type("DummyConfig", (), {"default_model": None, "models": {}})(),
    )

    process_a = process_api.SessionProcess(UUID(session_a.id))
    parts_a = [part async for part in process_a._encode_uploaded_files()]

    marker_a = session_a.dir / "uploads" / ".sent"
    assert any(shared_file.as_posix() in getattr(part, "text", "") for part in parts_a)
    assert json.loads(marker_a.read_text(encoding="utf-8")) == ["shared.txt"]

    restarted_process_a = process_api.SessionProcess(UUID(session_a.id))
    parts_a_restart = [part async for part in restarted_process_a._encode_uploaded_files()]
    assert parts_a_restart == []

    process_b = process_api.SessionProcess(UUID(session_b.id))
    parts_b = [part async for part in process_b._encode_uploaded_files()]

    marker_b = session_b.dir / "uploads" / ".sent"
    assert any(shared_file.as_posix() in getattr(part, "text", "") for part in parts_b)
    assert json.loads(marker_b.read_text(encoding="utf-8")) == ["shared.txt"]
