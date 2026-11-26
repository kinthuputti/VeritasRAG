"""Check on your own computer whether a password matches a hash from hash_password.py (nothing is sent anywhere):
    python tools/check_password.py
Paste the whole  scrypt$...$...  line when asked, then the password."""
import getpass, hashlib, hmac
stored = input("Hash line: ").strip().strip('"').strip("'")
pw = getpass.getpass("Password: ")
try:
    _, salt, h = stored.split("$")
    ok = hmac.compare_digest(hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex(), h)
except Exception as e:
    raise SystemExit(f"The hash line is damaged ({e.__class__.__name__}). It must be one line: scrypt$<32 characters>$<128 characters>")
print("MATCH: this password goes with this hash." if ok else "NO MATCH: wrong password, or the hash was copied wrongly.")
