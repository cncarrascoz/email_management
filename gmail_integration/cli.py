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


def cmd_list_labels(args):
    svc = get_gmail_service()
    results = svc.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])
    print(f"📦  {len(labels)} labels:")
    for lbl in labels:
        print(f" - {lbl['name']}")


def _decode_payload(part) -> str:
    """Decode RFC822 or base64 message part into plain text."""
    if "data" in part["body"]:
        data = part["body"]["data"]
        decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
        return decoded_bytes.decode("utf-8", errors="ignore")
    return ""


def _extract_plain_text(msg) -> str:
    """Return plain-text subject + snippet or body."""
    headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}
    subject = headers.get("subject", "")
    body = ""
    # Check if the message has a plain text part
    if msg["payload"]["mimeType"] == "text/plain":
        body = _decode_payload(msg["payload"])
    else:
        for part in msg["payload"].get("parts", []):
            if part["mimeType"] == "text/plain":
                body = _decode_payload(part)
                break
    return f"Subject: {subject}\n{body}"


def cmd_fetch(args):
    svc = get_gmail_service()
    res = svc.users().messages().list(userId="me", maxResults=args.max).execute()
    ids = [m["id"] for m in res.get("messages", [])]
    print(f"Fetched {len(ids)} message IDs. Downloading metadata...")

    emails = []
    for mid in ids:
        msg = svc.users().messages().get(userId="me", id=mid, format="full").execute()
        text = _extract_plain_text(msg)
        emails.append({"id": mid, "text": text})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json, io
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, ensure_ascii=False, indent=2)

    print(f"--- Saved {len(emails)} raw emails to {out_path}")


def main():
    ap = argparse.ArgumentParser(prog="gmail-cli", description="Gmail ML integration CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub_auth = sub.add_parser("auth", help="Authenticate & show account email")
    sub_auth.set_defaults(func=cmd_auth)

    sub_lbls = sub.add_parser("list-labels", help="List Gmail labels")
    sub_lbls.set_defaults(func=cmd_list_labels)

    sub_fetch = sub.add_parser("fetch", help="Fetch recent emails")
    sub_fetch.add_argument("--max", type=int, default=50, help="Number of emails")
    sub_fetch.add_argument("--out", default="out/emails.json")
    sub_fetch.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
