/// Build script: download the pre-built vectorlite SQLite extension library
/// for the current target platform from the official GitHub releases.
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let (lib_name, wheel_name) = match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "aarch64") => (
            "vectorlite.dylib",
            "vectorlite_py-0.2.0-py3-none-macosx_11_0_arm64.whl",
        ),
        ("macos", "x86_64") => (
            "vectorlite.dylib",
            "vectorlite_py-0.2.0-py3-none-macosx_10_15_x86_64.whl",
        ),
        ("linux", "x86_64") => (
            "vectorlite.so",
            "vectorlite_py-0.2.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        ),
        ("windows", "x86_64") => (
            "vectorlite.dll",
            "vectorlite_py-0.2.0-py3-none-win_amd64.whl",
        ),
        _ => {
            eprintln!(
                "cargo:warning=vectorlite: unsupported platform {target_os}/{target_arch}, skipping"
            );
            return;
        }
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join(lib_name);

    // Check if the library already exists in the project's .mempalace-bin directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let local_lib = manifest_dir.join(".mempalace-bin").join(lib_name);

    if local_lib.exists() {
        // Use the checked-in library
        if let Err(e) = fs::copy(&local_lib, &dest) {
            eprintln!("cargo:warning=vectorlite: failed to copy {local_lib:?} to {dest:?}: {e}");
        } else {
            println!("cargo:rustc-env=VECTORLITE_LIB_PATH={}", dest.display());
            println!("cargo:rerun-if-changed={}", local_lib.display());
            return;
        }
    }

    // Try to download from GitHub releases
    let url =
        format!("https://github.com/1yefuwang1/vectorlite/releases/download/v0.2.0/{wheel_name}");

    eprintln!("vectorlite: downloading {url}");

    let tmp_dir = out_dir.join("vectorlite_wheel");
    let _ = fs::create_dir_all(&tmp_dir);

    let wheel_path = tmp_dir.join(wheel_name);

    // Try curl first, then fall back to powershell on Windows
    if let Err(e) = download_with_curl(&url, &wheel_path) {
        if target_os == "windows" {
            if let Err(e2) = download_with_powershell(&url, &wheel_path) {
                eprintln!("cargo:warning=vectorlite: download failed (curl: {e}, pwsh: {e2})");
                return;
            }
        } else {
            eprintln!("cargo:warning=vectorlite: download failed: {e}");
            return;
        }
    }

    // Extract the shared library from the wheel (a wheel is a zip file)
    match extract_from_wheel(&wheel_path, &dest) {
        Ok(()) => {
            println!("cargo:rustc-env=VECTORLITE_LIB_PATH={}", dest.display());
            eprintln!("vectorlite: extracted to {}", dest.display());
        }
        Err(e) => {
            eprintln!("cargo:warning=vectorlite: extraction failed: {e}");
        }
    }
}

fn download_with_curl(url: &str, dest: &Path) -> io::Result<()> {
    let status = Command::new("curl")
        .args([
            "-sL",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "-o",
            dest.to_str().unwrap(),
            url,
        ])
        .status()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("curl not found: {e}")))?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("curl exit status: {status}")))
    }
}

fn download_with_powershell(url: &str, dest: &Path) -> io::Result<()> {
    let status = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                url,
                dest.display()
            ),
        ])
        .status()
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("powershell not found: {e}"),
            )
        })?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "powershell exit status: {status}"
        )))
    }
}

fn extract_from_wheel(wheel_path: &Path, dest: &Path) -> io::Result<()> {
    // A .whl file is a ZIP archive. We find the vectorlite.* file inside and extract it.
    let file = fs::File::open(wheel_path)?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let name = entry.name().to_owned();

        if name.contains("vectorlite.")
            && (name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll"))
        {
            let mut dest_file = fs::File::create(dest)?;
            io::copy(&mut entry, &mut dest_file)?;
            return Ok(());
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "vectorlite shared library not found in wheel",
    ))
}
