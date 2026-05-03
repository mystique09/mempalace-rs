# Cross-Platform Build CI Design

## Goal

Add CI that builds release binaries for the Rust workspace on Android and Windows without using a Windows runner.

## Scope

The CI should build the two shipped binaries:

- `mempalace-rs`
- `mempalace-mcp`

The CI should target:

- `aarch64-linux-android`
- `x86_64-pc-windows-gnu`

The workflow should run on `ubicloud-standard-4`.

## Approach

Create a dedicated GitHub Actions workflow for build validation and artifact upload. Keep the existing Android release workflow unchanged because it publishes rolling and tagged release artifacts to Cloudflare R2 and GitHub Releases.

Use a target matrix so Android and Windows share the same workflow structure while keeping target-specific setup steps explicit:

- Android installs Rust target support and the Android NDK, then configures the clang linker and archive tools.
- Windows installs Rust target support and the MinGW cross toolchain, then builds `x86_64-pc-windows-gnu` from Linux.

Both matrix jobs compile `mempalace-rs` and `mempalace-mcp-bin` in release mode, package the produced binaries, generate checksums, and upload the archives through GitHub Actions artifacts.

## Packaging

Android artifacts should be tarballs containing extensionless binaries:

- `mempalace-rs-aarch64-linux-android.tar.gz`
- `mempalace-mcp-aarch64-linux-android.tar.gz`

Windows artifacts should be zip files containing `.exe` binaries:

- `mempalace-rs-x86_64-pc-windows-gnu.zip`
- `mempalace-mcp-x86_64-pc-windows-gnu.zip`

Each target should include a `SHA256SUMS.txt` file in the uploaded artifact bundle.

## Triggers

Run on pull requests, pushes to `main`, and manual dispatch. This makes the workflow useful as CI without changing the existing release process.

## Error Handling

Let setup or build failures fail the workflow. Upload no partial artifact if the build or packaging step fails.

## Verification

Validate the workflow syntax locally enough to catch obvious YAML and shell issues. The full proof is the GitHub Actions run because the Android NDK and MinGW cross-compilation environment are CI-specific.
