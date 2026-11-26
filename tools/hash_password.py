"""Make a salted password hash for a staff account (no installs needed):  python tools/hash_password.py
Paste the printed line into Streamlit Secrets (see streamlit_demo/secrets.example.toml). The password itself is never stored."""
import getpass, hashlib, os
pw = getpass.getpass("Password: ")
if len(pw) < 10: raise SystemExit("Use at least 10 characters.")
salt = os.urandom(16).hex()
print("scrypt$" + salt + "$" + hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex())
