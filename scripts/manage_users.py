"""Manage password hashes locally; publish the resulting JSON as an HF Secret."""
import argparse
import getpass
import json
import os
from pathlib import Path

from rag.auth import hash_password, validate_users


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('username')
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--file', type=Path, default=Path('data/access-users.json'))
    args = parser.parse_args()
    users = validate_users(json.loads(args.file.read_text())) if args.file.exists() else {}
    if args.remove:
        if args.username not in users:
            parser.error('Unknown username.')
        if len(users) == 1:
            parser.error('Keep at least one account to avoid locking out the owner.')
        del users[args.username]
    else:
        password = getpass.getpass('New password (hidden): ')
        if password != getpass.getpass('Repeat password (hidden): '):
            parser.error('Passwords do not match.')
        users[args.username] = hash_password(password)
    validate_users(users)
    args.file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.file.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(users, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    temporary.replace(args.file)
    print(f'Saved {len(users)} account(s) to {args.file}. Password values were not saved.')


if __name__ == '__main__':
    main()
