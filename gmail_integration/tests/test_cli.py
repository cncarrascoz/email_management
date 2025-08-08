import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gmail_integration import cli


# Helper functions

# Run the CLI with given arguments, simulating command-line execution
def run_cli_with_args(args):
    """Invoke the CLI exactly as from the command-line."""
    with patch("sys.argv", ["cli.py"] + args):
        cli.main()

# Mock Gmail service to return deterministic messages
def _build_mock_service(num_msgs: int) -> MagicMock:
    """
    Return a mocked Gmail `service` that produces `num_msgs`
    deterministic plain-text messages.
    """
    svc = MagicMock()

    # service.users().messages()
    users = MagicMock()
    svc.users.return_value = users
    messages = MagicMock()
    users.messages.return_value = messages

    # list → IDs
    messages.list.return_value.execute.return_value = {
        "messages": [{"id": f"msg{i}"} for i in range(1, num_msgs + 1)]
    }

    # get → full message
    def _get_side_effect(userId, id, format):        
        idx = int(id.replace("msg", ""))
        subject = f"Test Subject {idx}"
        body = f"Test body {idx}"
        encoded = base64.urlsafe_b64encode(body.encode()).decode()

        call = MagicMock()
        call.execute.return_value = {
            "id": id,
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [{"name": "Subject", "value": subject}],
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": encoded}},
                ],
            },
        }
        return call

    messages.get.side_effect = _get_side_effect
    return svc


# Tests

# auth
@patch("gmail_integration.cli.get_gmail_service")
def test_auth_command_prints_email(mock_get_service, capsys):
    mock_svc = MagicMock()
    mock_svc.users().getProfile().execute.return_value = {
        "emailAddress": "test@example.com"
    }
    mock_get_service.return_value = mock_svc

    run_cli_with_args(["auth"])
    out = capsys.readouterr().out
    assert "✅ Authenticated! Gmail address: test@example.com" in out


# list-labels
@patch("gmail_integration.cli.get_gmail_service")
def test_list_labels_command(mock_get_service, capsys):
    mock_svc = MagicMock()
    mock_svc.users().labels().list().execute.return_value = {
        "labels": [
            {"id": "INBOX", "name": "INBOX"},
            {"id": "SENT", "name": "SENT"},
            {"id": "IMPORTANT", "name": "IMPORTANT"},
        ]
    }
    mock_get_service.return_value = mock_svc

    run_cli_with_args(["list-labels"])
    out = capsys.readouterr().out
    assert "📦  3 labels:" in out
    for lbl in ["INBOX", "SENT", "IMPORTANT"]:
        assert f" - {lbl}" in out


# fetch  (tests for both default & custom paths)
@pytest.mark.parametrize("num_msgs, use_custom_out", [(2, False), (1, True)])
def test_fetch_command(tmp_path, capsys, monkeypatch, num_msgs, use_custom_out):
    with patch(
        "gmail_integration.cli.get_gmail_service",
        return_value=_build_mock_service(num_msgs),
    ):
        cli_args = ["fetch", "--max", str(num_msgs)]

        if use_custom_out:
            custom = tmp_path / "custom_emails.json"
            cli_args += ["--out", str(custom)]
            expected_path = custom
        else:
            # Default path: out/emails.json
            monkeypatch.chdir(tmp_path)
            expected_path = Path("out/emails.json")

        run_cli_with_args(cli_args)

    # console output
    out = capsys.readouterr().out
    assert f"Fetched {num_msgs} message IDs" in out
    assert f"Saved {num_msgs} raw emails" in out

    # file contents
    saved = json.loads(expected_path.read_text(encoding="utf-8"))
    expected = [
        {"id": f"msg{i}", "text": f"Subject: Test Subject {i}\nTest body {i}"}
        for i in range(1, num_msgs + 1)
    ]
    assert saved == expected
