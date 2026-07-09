#!/usr/bin/env python3
"""Generate tumour_auth salt/hash for .streamlit/secrets.toml."""

import argparse
import getpass
import hashlib
import hmac
import secrets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", help="Password to hash (prompted if omitted)")
    args = parser.parse_args()
    password = args.password or getpass.getpass("Password: ")
    salt = secrets.token_hex(16)
    digest = hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    print("[tumor_auth]")
    print(f'salt = "{salt}"')
    print(f'hash = "{digest}"')


if __name__ == "__main__":
    main()
