# Cloudflare R2 release artifact requirements

This repository's Android release workflow at `.github/workflows/release-to-prod.yml` now uploads packaged build artifacts to Cloudflare R2 instead of using the GitHub Actions artifact store.

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

- the runner has the `aws` CLI available in `PATH`
- the provided R2 access key can write to the configured bucket
- the bucket and any optional public domain are already configured on the Cloudflare side

## Uploaded files

The workflow uploads these files from `dist/`:

- `mempalace-rs-<target>.tar.gz`
- `mempalace-mcp-<target>.tar.gz`
- `SHA256SUMS.txt`

## R2 object layout

Artifacts are written under target-specific prefixes:

- commit uploads: `mempalace/android/<target>/commits/<git-sha>/`
- rolling uploads from `main`: `mempalace/android/<target>/rolling/`
- tagged uploads: `mempalace/android/<target>/tags/<tag-name>/`

For the current workflow target, `<target>` is `aarch64-linux-android`.

## Workflow behavior

- Pushes to `main` upload to the commit prefix and the rolling prefix.
- Tag pushes upload to the commit prefix and the matching tag prefix.
- `workflow_dispatch` uploads to the commit prefix for the ref the workflow runs against.
- The GitHub Actions step summary includes the R2 prefixes, and public URLs when `CLOUDFLARE_R2_PUBLIC_BASE_URL` is set.

## Scope of this change

This change replaces the `actions/upload-artifact` step only.

The workflow still publishes GitHub release assets through the existing `softprops/action-gh-release` steps.
