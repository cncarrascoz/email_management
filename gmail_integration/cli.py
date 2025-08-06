"""
Lightweight CLI for Gmail integration tasks:
  • auth                 : trigger OAuth flow / verify login

Run:
    python gmail_integration/cli.py auth
"""

import argparse
import base64
import email
from pathlib import Path

from gmail_integration.auth.authorize import get_gmail_service


def cmd_auth(args):
    svc = get_gmail_service()
    profile = svc.users().getProfile(userId="me").execute()
    print(f"✅ Authenticated! Gmail address: {profile['emailAddress']}")



def main():
    ap = argparse.ArgumentParser(prog="gmail-cli", description="Gmail ML integration CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub_auth = sub.add_parser("auth", help="Authenticate & show account email")
    sub_auth.set_defaults(func=cmd_auth)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
