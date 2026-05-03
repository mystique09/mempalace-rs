# Cross-Platform Build CI Design

## Goal

Add CI that builds release binaries for the Rust workspace on Android and Windows without using a Windows runner, then attaches those binaries to GitHub releases.

## Scope

The CI should build the two shipped binaries:

- `mempalace-rs`
- `mempalace-mcp`

The CI should target:

- `aarch64-linux-android`
- `x86_64-pc-windows-gnu`

The workflow should run on `ubicloud-standard-4`.

## Approach

Extend the existing release workflow because it already publishes rolling and tagged release artifacts to Cloudflare R2 and GitHub Releases.

Use a target matrix so Android and Windows share the same workflow structure while keeping target-specific setup steps explicit:

- Android installs Rust target support and the Android NDK, then configures the clang linker and archive tools.
- Windows installs Rust target support and the MinGW cross toolchain, then builds `x86_64-pc-windows-gnu` from Linux.

Both matrix jobs compile `mempalace-rs` and `mempalace-mcp-bin` in release mode, package the produced binaries, generate checksums, and upload the archives through GitHub Actions artifacts.

A release job should download the packaged matrix artifacts, generate an aggregate checksum file, update the rolling release on pushes to `main`, and attach all Android and Windows packages to rolling and tagged GitHub releases.

## Packaging

Android artifacts should be tarballs containing extensionless binaries:

- `mempalace-rs-aarch64-linux-android.tar.gz`
- `mempalace-mcp-aarch64-linux-android.tar.gz`

Windows artifacts should be zip files containing `.exe` binaries:

- `mempalace-rs-x86_64-pc-windows-gnu.zip`
- `mempalace-mcp-x86_64-pc-windows-gnu.zip`

Each target should include a target-specific checksum file in the uploaded artifact bundle. The release job should also publish an aggregate `SHA256SUMS.txt`.

## Triggers

Run on pull requests, pushes to `main`, tags matching `v*`, and manual dispatch. Pull requests should build and upload GitHub Actions artifacts only. Pushes to `main` and tags should also publish release assets.

## Error Handling

Let setup or build failures fail the workflow. Upload no partial artifact if the build or packaging step fails.

## Verification

Validate the workflow syntax locally enough to catch obvious YAML and shell issues. The full proof is the GitHub Actions run because the Android NDK and MinGW cross-compilation environment are CI-specific.
