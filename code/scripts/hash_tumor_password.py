#!/usr/bin/env python3
"""Generate atlas auth salt/hash for .streamlit/secrets.toml."""

import argparse
import getpass
import hashlib
import hmac
import secrets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", help="Password to hash (prompted if omitted)")
    parser.add_argument(
        "--section",
        default="tumor_auth",
        choices=("tumor_auth", "other_auth"),
        help="Secrets section to print (default: tumor_auth)",
    )
    parser.add_argument(
        "--salt",
        help="Reuse an existing salt when adding extra hashes (otherwise a new salt is generated)",
    )
    args = parser.parse_args()
    password = args.password or getpass.getpass("Password: ")
    salt = args.salt or secrets.token_hex(16)
    digest = hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    print(f"[{args.section}]")
    print(f'salt = "{salt}"')
    print(f'hash = "{digest}"')


if __name__ == "__main__":
    main()
