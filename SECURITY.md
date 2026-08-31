# Security policy

## Supported versions

Only the latest published release and the default branch are expected to
receive security fixes. This is a research platform, not a production service.

## Reporting a vulnerability

Please do not open a public issue for credentials, private data, malicious
archives, or an exploitable dependency. Once the repository is published,
use GitHub's private vulnerability reporting channel or contact the
maintainers privately through the repository profile.

Include the affected file/version, reproduction steps, impact, and a safe
minimal proof of concept. Do not include real secrets or private datasets.

The fetchers handle external archives and validate archive paths before
extraction, but users should still run them in an isolated environment and
review downloaded sources before use.
