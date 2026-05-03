# Cross-Platform Release CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mempalace-rs` and `mempalace-mcp` release binaries for Android and Windows on `ubicloud-standard-4`, then attach the packages to rolling and tagged GitHub releases.

**Architecture:** Convert the existing release workflow into a build matrix with Android and Windows targets. Matrix jobs package target-specific binaries and upload workflow artifacts; a separate release job aggregates those artifacts and publishes them to GitHub Releases.

**Tech Stack:** GitHub Actions, Rust stable toolchain, Android NDK, MinGW GNU cross toolchain, Cloudflare R2 S3 uploads, `softprops/action-gh-release`.

---

### Task 1: Update The Release Workflow

**Files:**
- Modify: `.github/workflows/release-to-prod.yml`

- [ ] **Step 1: Replace the single-target workflow with a matrix**

Use this structure:

```yaml
name: Build and Release Binaries

on:
  pull_request:
  push:
    branches: [main]
    tags:
      - "v*"
  workflow_dispatch:
```

Keep `runs-on: ubicloud-standard-4` for every build job.

- [ ] **Step 2: Build both binaries for both targets**

Run this cargo command in both matrix jobs:

```bash
cargo build --release --timings \
  --target "$TARGET" \
  --package mempalace-rs \
  --package mempalace-mcp-bin
```

For Android, configure the NDK clang linker. For Windows, install MinGW and set the GNU linker to `x86_64-w64-mingw32-gcc`.

- [ ] **Step 3: Package target-specific artifacts**

Create these files:

```text
dist/out/mempalace-rs-aarch64-linux-android.tar.gz
dist/out/mempalace-mcp-aarch64-linux-android.tar.gz
dist/out/SHA256SUMS-aarch64-linux-android.txt
dist/out/mempalace-rs-x86_64-pc-windows-gnu.zip
dist/out/mempalace-mcp-x86_64-pc-windows-gnu.zip
dist/out/SHA256SUMS-x86_64-pc-windows-gnu.txt
```

- [ ] **Step 4: Upload matrix artifacts**

Use `actions/upload-artifact@v4` with names like:

```text
release-binaries-aarch64-linux-android
release-binaries-x86_64-pc-windows-gnu
```

- [ ] **Step 5: Publish release assets once**

Add a release job that depends on the matrix build, downloads all `release-binaries-*` artifacts, writes aggregate `dist/SHA256SUMS.txt`, updates the rolling release on pushes to `main`, and attaches `dist/*` to both rolling and tagged GitHub releases.

### Task 2: Update Release Artifact Documentation

**Files:**
- Modify: `docs/cloudflare-r2-release-artifacts.md`

- [ ] **Step 1: Update target and file descriptions**

Document Android tarballs, Windows zip files, per-target checksum files, and aggregate GitHub release checksums.

- [ ] **Step 2: Update R2 object layout**

Document target prefixes as:

```text
mempalace/<platform>/<target>/commits/<git-sha>/
mempalace/<platform>/<target>/rolling/
mempalace/<platform>/<target>/tags/<tag-name>/
```

### Task 3: Verify The Change

**Files:**
- Read: `.github/workflows/release-to-prod.yml`
- Read: `docs/cloudflare-r2-release-artifacts.md`

- [ ] **Step 1: Parse the workflow as YAML**

Run:

```bash
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/release-to-prod.yml"); puts "workflow yaml ok"'
```

Expected output:

```text
workflow yaml ok
```

- [ ] **Step 2: Check changed files**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; status lists only the intended workflow and docs changes.
