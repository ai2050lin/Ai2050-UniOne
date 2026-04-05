param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$targets = @(
    "tempdata",
    "tests/codex_temp",
    "nfb_data"
)

Write-Host "Repository root: $repoRoot"
Write-Host "Targets to untrack from git index (local files are kept):"
$targets | ForEach-Object { Write-Host "  - $_" }

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only. Planned command:"
    Write-Host "git rm -r --cached --ignore-unmatch -- $($targets -join ' ')"
    exit 0
}

git rm -r --cached --ignore-unmatch -- $targets

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Review with: git status --short"
Write-Host "  2. Commit the index cleanup and .gitignore update"
Write-Host "  3. Push again"
