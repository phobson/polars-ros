# Puts the shell into a state where `cargo`/`rustc` can build this crate.
#
# Two machine-specific problems are worked around here:
#
# 1. Rust 1.99.0 is blocked by a Defender Exploit Guard "block process image"
#    rule, regardless of where it is installed (a fresh RUSTUP_HOME under
#    C:\Users\phobson\.rustup-alt does not help). 1.95.0 is not blocked and is
#    new enough for the dependency graph: `sysinfo` 0.39 requires rustc >= 1.95.
#    The 1.95.0 toolchain bin directory goes first on PATH so the rustup proxy
#    -- which would dispatch to the blocked default toolchain -- is never used.
#
# 2. There is no Visual Studio C++ build tools install and no Windows 10/11 SDK.
#    Visual Studio 2022 Community does ship a VS2017-era toolset and a matching
#    SDK under SDK\ScopeCppSDK\vc15, which is what `cl.exe` needs to compile the
#    C shims in `stacker` and `zstd`.
#
# Usage:  . tools\msvc-env.ps1      (dot-source, so the env changes stick)

$toolchain = 'C:\Users\phobson\.rustup-alt\toolchains\1.95.0-x86_64-pc-windows-msvc\bin'
$vc15      = 'C:\Program Files\Microsoft Visual Studio\2022\Community\SDK\ScopeCppSDK\vc15'

$env:RUSTUP_HOME = 'C:\Users\phobson\.rustup-alt'
$env:CARGO_HOME  = 'C:\Users\phobson\.cargo-alt'
# Stop rustup from helpfully auto-installing the blocked default toolchain.
$env:RUSTUP_AUTO_INSTALL = '0'

$env:PATH = "$toolchain;$vc15\VC\bin;$vc15\SDK\bin;$env:PATH"

$env:INCLUDE = "$vc15\VC\include;$vc15\SDK\include\ucrt;$vc15\SDK\include\um;$vc15\SDK\include\shared"
$env:LIB     = "$vc15\VC\lib;$vc15\SDK\lib"

Write-Host "rustc $(& rustc --version)"
Write-Host "cargo $(& cargo --version)"