"""Sign-in helpers: salted password hashes for people accounts, and verification of identity-provider tokens (OIDC / JWT).

The identity provider (Google, Microsoft Entra ID, Okta...) does the actual login, including two-factor authentication.
This code only checks the signed token it hands back, then turns the user's groups into roles."""
import hashlib, hmac, os

def hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex()
    return f"scrypt${salt}${h}"

def check_password(password, stored):
    try: _, salt, h = stored.split("$")
    except ValueError: return False
    return hmac.compare_digest(hash_password(password, salt).split("$")[2], h)

def parse_group_map(raw):
    """'hr-team=hr;finance-team=finance;it-admins=admin|uploader' -> {group: [roles]}"""
    out = {}
    for part in (raw or "").split(";"):
        if "=" in part:
            g, r = part.split("=", 1); out[g.strip()] = [x.strip() for x in r.split("|") if x.strip()]
    return out

class JwtVerifier:
    """Verifies a bearer token: signature, expiry (required), issuer and audience when configured.
    Give exactly one of: secret (HS256), public_key PEM (RS256) or jwks_url (RS256 keys fetched from the provider)."""
    def __init__(self, secret=None, public_key=None, jwks_url=None, audience=None, issuer=None, group_map=None,
                 group_claim="groups", user_claim="email"):
        if sum(x is not None for x in (secret, public_key, jwks_url)) != 1:
            raise ValueError("configure exactly one of secret, public_key, jwks_url")
        self.secret, self.public_key, self.audience, self.issuer = secret, public_key, audience, issuer
        self.algorithms = ["HS256"] if secret else ["RS256"]
        self.group_map, self.group_claim, self.user_claim = group_map or {}, group_claim, user_claim
        self._jwks = None
        if jwks_url:
            import jwt
            self._jwks = jwt.PyJWKClient(jwks_url)

    def verify(self, token):
        import jwt
        try:
            key = self.secret or self.public_key
            if self._jwks: key = self._jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=self.algorithms, audience=self.audience, issuer=self.issuer,
                                options={"require": ["exp"], "verify_aud": self.audience is not None})
        except Exception as e:                      # bad signature, expired, wrong audience, malformed...
            raise ValueError(f"invalid token: {e.__class__.__name__}")
        user = claims.get(self.user_claim) or claims.get("sub")
        if not user: raise ValueError("token has no user")
        roles = {"employee"}                        # being signed in at all means employee
        for g in claims.get(self.group_claim, []) or []: roles.update(self.group_map.get(g, []))
        return dict(user=user, roles=sorted(roles), expires=None)
