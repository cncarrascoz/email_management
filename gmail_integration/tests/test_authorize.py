import pytest
from unittest.mock import MagicMock, patch
from gmail_integration.auth.authorize import get_gmail_service

@patch("gmail_integration.auth.authorize.build")
@patch("gmail_integration.auth.authorize._load_saved_creds")
def test_get_gmail_service_with_valid_creds(mock_load_creds, mock_build):
    mock_creds = MagicMock()
    mock_load_creds.return_value = mock_creds
    mock_load_creds.return_value = mock_creds

    mock_service = MagicMock()
    mock_build.return_value = mock_service

    service = get_gmail_service()
    assert service == mock_service
    mock_build.assert_called_once_with("gmail", "v1", credentials=mock_creds, cache_discovery=False)
