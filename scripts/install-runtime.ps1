# 装项目自带的 Python 运行时。
#
# 为什么用 PowerShell 写这个：电脑上完全没有 Python 的时候，也得有东西能把
# 运行时下载下来——PowerShell 是 Windows 自带的，不用装任何东西。
#
# 只做三件事：读 runtime.lock.json -> 下载并校验 sha256 -> 解压到 runtime\python。
# 版本、文件名、校验值都在 runtime.lock.json 里，不在这个脚本里重复。
#
# 注意：本文件必须存成「UTF-8 带 BOM」。Windows PowerShell 5.1 读无 BOM 的
# UTF-8 会按 ANSI 解码，中文注释会变成乱码并导致语法错误。

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$lockPath = Join-Path $root 'runtime.lock.json'
$runtimeDir = Join-Path $root 'runtime\python'
$cacheDir = Join-Path $root 'runtime\_cache'
$stagingDir = Join-Path $root 'runtime\_staging'
$markerPath = Join-Path $runtimeDir '.selp-asr-runtime.json'

function Say($text) { Write-Host $text }

function Fail($text) {
    Write-Host ''
    Write-Host '  [install failed]'
    Write-Host $text
    Write-Host ''
    exit 1
}

if (-not (Test-Path $lockPath)) { Fail "Missing runtime.lock.json. Please download the complete project." }

try {
    $lock = Get-Content $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    Fail "runtime.lock.json is corrupted: $($_.Exception.Message)"
}

# Pick the asset for this machine.
$machine = $env:PROCESSOR_ARCHITECTURE
if ($machine -eq 'AMD64') { $key = 'windows-amd64' }
elseif ($machine -eq 'ARM64') { $key = 'windows-arm64' }
else { Fail "Unsupported CPU architecture: $machine" }

$entry = $lock.platforms.PSObject.Properties[$key].Value
if (-not $entry) { Fail "runtime.lock.json has no entry for $key" }

$assetName = $entry.asset
$expected = [int64]$entry.size
$expectedHash = $entry.sha256

New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$archive = Join-Path $cacheDir $assetName

function Get-Sha256($path) {
    # Get-FileHash comes from a PowerShell module; if autoloading is broken (a
    # redirected PSModulePath is enough), fall back to the .NET API directly.
    $cmd = Get-Command Get-FileHash -ErrorAction SilentlyContinue
    if ($cmd) { return (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower() }

    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($path)
    try {
        $hashBytes = $sha.ComputeHash($stream)
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
    return ([BitConverter]::ToString($hashBytes) -replace '-', '').ToLower()
}

function Get-Human($bytes) {
    if ($bytes -ge 1GB) { return '{0:N1} GB' -f ($bytes / 1GB) }
    if ($bytes -ge 1MB) { return '{0:N1} MB' -f ($bytes / 1MB) }
    return '{0:N0} KB' -f ($bytes / 1KB)
}

# A cached file is reused only if its hash matches.
if (Test-Path $archive) {
    if ((Get-Sha256 $archive) -ne $expectedHash) { Remove-Item $archive -Force }
}

if (-not (Test-Path $archive)) {
    $release = $entry.release
    if (-not $release) { $release = $lock.release }

    $ghTpl = $lock.sources.github
    $njuTpl = $lock.sources.nju
    $ghUrl = if ($ghTpl) { ($ghTpl -replace '\{release\}', $release) + '/' + $assetName }
    $njuUrl = if ($njuTpl) { ($njuTpl -replace '\{release\}', $release) + '/' + $assetName }
    if (-not $ghUrl -and -not $njuUrl) { Fail 'runtime.lock.json has incomplete sources' }

    $hasCurl = $null -ne (Get-Command curl.exe -ErrorAction SilentlyContinue)

    # Order the sources by measured speed. Only possible with curl (it reports
    # speed_download); without it, mirror-first is still a safe default.
    $candidates = @()
    foreach ($u in @($njuUrl, $ghUrl)) {
        if ($u) { $candidates += $u }
    }

    if ($hasCurl -and $candidates.Count -gt 1) {
        Say '  testing download sources ...'
        $scored = @()
        foreach ($u in $candidates) {
            $speed = & curl.exe -sS -o NUL --max-time 8 -r 0-1048575 -w '%{speed_download}' $u 2>$null
            $value = 0
            [void][double]::TryParse("$speed", [ref]$value)
            $name = ([Uri]$u).Host
            if ($value -gt 0) { Say ("    {0,-32} {1:N0} KB/s" -f $name, ($value / 1KB)) }
            else { Say ("    {0,-32} unreachable" -f $name) }
            $scored += [pscustomobject]@{ Speed = $value; Url = $u }
        }
        $candidates = @($scored | Sort-Object -Property Speed -Descending | ForEach-Object { $_.Url })
    }

    $urls = $candidates

    Say ''
    Say "  Bundled Python $($lock.python) -> runtime\python"
    Say '  (private to this project, your own Python is not touched)'
    Say ("  downloading " + (Get-Human $expected))

    $done = $false

    for ($i = 0; $i -lt $urls.Count; $i++) {
        $url = $urls[$i]
        $hostName = ([Uri]$url).Host
        Say "    source $($i + 1)/$($urls.Count): $hostName"
        try {
            if ($hasCurl) {
                & curl.exe -sS -fL --retry 2 --retry-delay 1 --connect-timeout 20 -o $archive $url
                if ($LASTEXITCODE -ne 0) { throw "curl exited with $LASTEXITCODE" }
            } else {
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -Uri $url -OutFile $archive -TimeoutSec 300 -UseBasicParsing
            }
            if (-not (Test-Path $archive)) { throw 'no file after download' }
            $size = (Get-Item $archive).Length
            if ($size -ne $expected) { throw "size mismatch: expected $expected, got $size" }
            $hash = Get-Sha256 $archive
            if ($hash -ne $expectedHash) {
                Remove-Item $archive -Force
                throw "sha256 mismatch ($($hash.Substring(0,12))...)"
            }
            $done = $true
            break
        } catch {
            Say "      failed: $($_.Exception.Message)"
            if (Test-Path $archive) { Remove-Item $archive -Force }
        }
    }

    if (-not $done) {
        $lines = @(
            'Could not download the Python runtime. This project needs it to run.',
            '  1) Check that GitHub or the mirror is reachable from this network.',
            '  2) Download it manually, put it into runtime\_cache\, then run start-server.bat again.',
            "     file name must be: $assetName",
            "     $($urls[0])"
        )
        Fail ($lines -join [Environment]::NewLine)
    }
}
Say '  extracting ...'
if (Test-Path $stagingDir) { Remove-Item $stagingDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tar) { Fail 'tar.exe not found (needs Windows 10 1803+). Install on another PC and copy the whole project folder.' }
& tar.exe -xzf $archive -C $stagingDir
if ($LASTEXITCODE -ne 0) { Fail "extraction failed (tar exit $LASTEXITCODE)" }

# The archive has a single top-level "python" folder; strip it.
$inner = Join-Path $stagingDir 'python'
if (-not (Test-Path (Join-Path $inner 'python.exe'))) { Fail 'python.exe not found after extraction, archive may be incomplete' }

if (Test-Path $runtimeDir) { Remove-Item $runtimeDir -Recurse -Force }
Move-Item $inner $runtimeDir
Remove-Item $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $archive -Force -ErrorAction SilentlyContinue

# Marker file: the Python side compares it with runtime.lock.json.
$marker = [ordered]@{
    python = $lock.python
    sha256 = $expectedHash
    asset  = $assetName
}
$json = $marker | ConvertTo-Json
[IO.File]::WriteAllText($markerPath, $json, (New-Object Text.UTF8Encoding($false)))

Say "  runtime ready: $runtimeDir\python.exe"
exit 0
