"""
Handles Gmail OAuth 2.0 flow and returns an authenticated
googleapiclient.discovery.Resource object.

Usage (internal):
    from gmail_integration.auth.authorize import get_gmail_service
    service = get_gmail_service()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
# scopes needed to request to access Google APIs
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

ROOT = Path(__file__).resolve().parent  # where this file currently lives
TOKEN_PATH = ROOT / "token.json"
CREDS_PATH = ROOT / "credentials.json"


def _load_saved_creds(token_path: Path) -> Optional[Credentials]:
    """Return stored credentials if present + valid, else ret None."""
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            return creds
    return None


def _interactive_login(creds_path: Path, token_path: Path) -> Credentials:
    """Run local-server OAuth flow and save token."""
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    token_path.write_text(creds.to_json())
    return creds


def get_gmail_service(
    creds_path: Path | str = CREDS_PATH,
    token_path: Path | str = TOKEN_PATH,
):
    """
    Returns: googleapiclient.discovery.Resource (Gmail service)
    """
    creds_path = Path(creds_path)
    token_path = Path(token_path)

    creds = _load_saved_creds(token_path)
    if not creds:
        if not creds_path.exists():
            raise FileNotFoundError(
                f"Google OAuth client secrets not found at {creds_path}. "
                "Download 'credentials.json' from Google Cloud Console and place it there."
            )
        creds = _interactive_login(creds_path, token_path)

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service
