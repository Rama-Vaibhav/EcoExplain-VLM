"""Google Earth Engine Drive export + Google Drive download helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import ee

from .export_utils import PATCH_SUFFIXES, organize_inbox_tiffs, patch_tif_path


def start_drive_exports(
    exports: Iterable[tuple[str, ee.Image, int]],
    *,
    drive_folder: str,
    geom: ee.Geometry,
) -> list[ee.batch.Task]:
    """Queue Earth Engine Export.image.toDrive tasks."""
    tasks: list[ee.batch.Task] = []
    for prefix, image, scale in exports:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=prefix[:100],
            folder=drive_folder,
            fileNamePrefix=prefix,
            region=geom,
            scale=scale,
            maxPixels=1e8,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append(task)
    return tasks


def poll_ee_tasks(tasks: list[ee.batch.Task], *, poll_seconds: int = 30, timeout_minutes: int = 180) -> dict:
    """Wait until all EE tasks finish or timeout."""
    deadline = time.time() + timeout_minutes * 60
    states: dict[str, str] = {}
    while time.time() < deadline:
        states = {t.id: t.status().get("state", "UNKNOWN") for t in tasks}
        if all(s in {"COMPLETED", "FAILED", "CANCELLED"} for s in states.values()):
            break
        time.sleep(poll_seconds)
    return states


def save_export_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def download_drive_folder(
    folder_name: str,
    dest_dir: Path,
    *,
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> list[Path]:
    """Download all GeoTIFF files from a Google Drive folder via Drive API v3.

    First-time setup:
      1. Create OAuth credentials in Google Cloud Console (Desktop app).
      2. Save as ``credentials.json`` in the project root.
      3. Run this function once; a browser opens for consent.
      4. Token is cached in ``token.json``.

    Requires: google-api-python-client, google-auth-oauthlib
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError(
            "Install Drive download deps: pip install google-api-python-client google-auth-oauthlib"
        ) from exc

    import io

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    project_root = dest_dir.parents[1] if dest_dir.name == "raw_tiffs" else dest_dir.parent
    creds_file = credentials_path or (project_root / "credentials.json")
    token_file = token_path or (project_root / "token.json")

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_file.exists():
                raise FileNotFoundError(
                    f"Missing {creds_file}. Download OAuth Desktop credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find folder id by name (first match)
    folder_resp = (
        service.files()
        .list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name)",
            pageSize=5,
        )
        .execute()
    )
    folders = folder_resp.get("files", [])
    if not folders:
        raise FileNotFoundError(f"Google Drive folder not found: {folder_name}")
    folder_id = folders[0]["id"]

    downloaded: list[Path] = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        for item in resp.get("files", []):
            name = item["name"]
            if not name.lower().endswith((".tif", ".tiff", ".geotiff")):
                continue
            out_path = dest_dir / name
            if out_path.exists():
                downloaded.append(out_path)
                continue
            request = service.files().get_media(fileId=item["id"])
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            out_path.write_bytes(buffer.getvalue())
            downloaded.append(out_path)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return downloaded


def sync_from_drive_inbox(inbox_dir: Path, raw_dir: Path) -> dict:
    """Organize manually downloaded Drive files from data/drive_inbox/."""
    moved = organize_inbox_tiffs(inbox_dir, raw_dir)
    return {"organized": len(moved), "paths": [str(p) for p in moved]}


def expected_drive_filenames(patch_ids: Iterable[str]) -> list[str]:
    names: list[str] = []
    for pid in patch_ids:
        for suffix in PATCH_SUFFIXES:
            names.append(f"{pid}{suffix}.tif")
    return names


def verify_downloaded(raw_dir: Path, patch_ids: Iterable[str]) -> dict:
    missing = []
    for pid in patch_ids:
        for suffix in PATCH_SUFFIXES:
            if not patch_tif_path(raw_dir, pid, suffix).exists():
                missing.append(f"{pid}{suffix}.tif")
    return {"complete": not missing, "missing_files": missing, "missing_count": len(missing)}
