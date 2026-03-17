$logPath = "d:\develop\TransformerLens-main\tests\codex_temp\glm4_9b_resumable_download_20260318.log"
$repoRoot = "d:\develop\TransformerLens-main"

if (Test-Path $logPath) {
    try {
        Remove-Item $logPath -Force -ErrorAction Stop
    }
    catch {
    }
}

$command = @(
    "cd /d $repoRoot",
    "set HF_ENDPOINT=https://huggingface.co",
    "python -u tests/codex_temp/resumable_hf_snapshot_download.py --repo-id zai-org/GLM-4-9B-Chat-HF --cache-dir d:\develop\model\hub --endpoint https://huggingface.co --max-workers 1 --stall-timeout 240 --poll-interval 10 --status-interval 30 --retry-cooldown 20 --max-attempts 10 >> `"$logPath`" 2>&1"
) -join " && "

$process = Start-Process `
    -FilePath cmd.exe `
    -ArgumentList "/c", $command `
    -WindowStyle Hidden `
    -PassThru

$process | Select-Object Id, ProcessName, StartTime
