# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's private vulnerability
reporting feature for this repository. Do not include credentials, customer
data, or exploit details in a public issue.

## Deployment guidance

- Store provider credentials and database secrets outside Git.
- Set a stable `AKA_MODEL_SECRET_KEY` anywhere encrypted model settings are
  shared between processes.
- Put public deployments behind HTTPS, authentication, request-size limits,
  and rate limiting.
- Restrict database and connector accounts to the minimum required access.
- Review logs, traces, uploads, conversations, and generated indexes under a
  defined retention policy.
- Rotate any credential that has appeared in a commit, build log, screenshot,
  or shared archive.
