# Cloudflare R2 release artifact requirements

This repository's release workflow at `.github/workflows/release-to-prod.yml` uploads packaged build artifacts to Cloudflare R2 and attaches the same packages to GitHub releases.

## Required GitHub secrets

Configure these repository secrets before running the workflow:

- `CLOUDFLARE_R2_ACCESS_KEY_ID`
- `CLOUDFLARE_R2_SECRET_ACCESS_KEY`
- `CLOUDFLARE_R2_ACCOUNT_ID`
- `CLOUDFLARE_R2_BUCKET`

## Optional GitHub repository variable

Set this variable if you want the workflow summary to include public download URLs:

- `CLOUDFLARE_R2_PUBLIC_BASE_URL`

Example:

- `https://downloads.example.com`

## Runtime requirements

The workflow currently assumes:

- the runner can run `aws` from `PATH` or install `awscli` with `apt-get`
- the provided R2 access key can write to the configured bucket
- the bucket and any optional public domain are already configured on the Cloudflare side

## Uploaded files

The workflow uploads target-specific files from `dist/out/`.

Android packages are tarballs:

- `mempalace-rs-aarch64-linux-android.tar.gz`
- `mempalace-mcp-aarch64-linux-android.tar.gz`
- `SHA256SUMS-aarch64-linux-android.txt`

Windows packages are zip files:

- `mempalace-rs-x86_64-pc-windows-gnu.zip`
- `mempalace-mcp-x86_64-pc-windows-gnu.zip`
- `SHA256SUMS-x86_64-pc-windows-gnu.txt`

The GitHub release job also generates and attaches an aggregate `SHA256SUMS.txt`.

## R2 object layout

Artifacts are written under platform and target-specific prefixes:

- commit uploads: `mempalace/<platform>/<target>/commits/<git-sha>/`
- rolling uploads from `main`: `mempalace/<platform>/<target>/rolling/`
- tagged uploads: `mempalace/<platform>/<target>/tags/<tag-name>/`

Current workflow targets are:

- `mempalace/android/aarch64-linux-android/`
- `mempalace/windows/x86_64-pc-windows-gnu/`

## Workflow behavior

- Pushes to `main` upload to the commit prefix and the rolling prefix.
- Tag pushes upload to the commit prefix and the matching tag prefix.
- `workflow_dispatch` uploads to the commit prefix for the ref the workflow runs against, and publishes GitHub release assets only when that ref is `main` or a `v*` tag.
- Pull requests build both targets and upload GitHub Actions artifacts, but skip Cloudflare R2 and GitHub release publishing.
- The GitHub Actions step summary includes the R2 prefixes, and public URL bases when `CLOUDFLARE_R2_PUBLIC_BASE_URL` is set.
- The rolling GitHub release and tagged GitHub releases include both CLI and MCP packages for Android and Windows.
