import pytest
from unittest.mock import patch, MagicMock
from gmail_integration import cli

@patch("gmail_integration.cli.get_gmail_service")
def test_auth_command_prints_email(mock_get_service, capsys):
    mock_service = MagicMock()
    mock_service.users().getProfile().execute.return_value = {
        "emailAddress": "test@example.com"
    }
    mock_get_service.return_value = mock_service

    test_args = ["auth"]
    with patch("sys.argv", ["cli.py"] + test_args):
        cli.main()

    captured = capsys.readouterr()
    print(captured.out)
    assert "✅ Authenticated! Gmail address: test@example.com" in captured.out
