# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report privately through GitHub: **Security → Report a vulnerability** on
[Moissanite-Labs/heedle](https://github.com/Moissanite-Labs/heedle/security/advisories/new).

Include what you found, how to reproduce it, and what an attacker could do with it.

## Scope

In scope:

- Code in `heedle/`, `web/` and `tools/`.
- Ways a draft can pass the verifier in `heedle/draft.py` while breaking a rule in [docs/protocol.md](docs/protocol.md). A forged draft that verifies is a security bug here, not only a correctness bug.
- Ways the vendored gate in `web/vendor/` can be made to differ from weft's.

Out of scope:

- The instruments prove nothing about who used them. A draft records the gesture the page saw. This is stated in the protocol and is not a vulnerability.
- The hosts are toys. That a toy model can be steered into saying something is not a finding.
