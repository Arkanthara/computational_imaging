param(
    [ValidateSet("build", "watch", "clean")]
    [string]$Command = "build",
    [ValidateSet("auto", "tinymist", "typst", "none")]
    [string]$PreviewEngine = "auto",
    [switch]$NoCache,
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$reportDir = (Resolve-Path $PSScriptRoot).Path
$repoRoot = (Resolve-Path (Join-Path $reportDir "..")).Path
$codeProject = Join-Path $repoRoot "code"
$submoduleProject = Join-Path $reportDir "typst_pyexec"
$sourceFile = Join-Path $reportDir "report.typ"

$uvArgs = @(
    "--project", $codeProject,
    "run",
    "--with-editable", $submoduleProject,
    "typst_pyexec",
    $Command
)

if ($Command -ne "clean") {
    $uvArgs += $sourceFile
    # $uvArgs += "--typst-compile-arg=--root"
    # $uvArgs += "--typst-compile-arg=$repoRoot"
}

if ($Command -eq "watch") {
    $uvArgs += "--preview-engine"
    $uvArgs += $PreviewEngine
    # $uvArgs += "--typst-watch-arg=--root"
    # $uvArgs += "--typst-watch-arg=$repoRoot"
}

if ($NoCache) {
    $uvArgs += "--no-cache"
}

if ($ExtraArgs) {
    $uvArgs += $ExtraArgs
}

Write-Host "Running: uv $($uvArgs -join ' ')"
& uv @uvArgs
exit $LASTEXITCODE
