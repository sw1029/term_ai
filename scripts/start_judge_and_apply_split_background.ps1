param(
    [ValidateSet("train", "dev", "test")]
    [string]$Split = "dev",
    [string]$RunScript = ".\scripts\run_judge_and_apply_split.ps1",
    [string]$JudgeModel = "gpt-5.5",
    [string]$JudgeReasoningEffort = "low",
    [string]$GeneratorModel = "gpt-5.4-mini",
    [double]$RequestsPerSecond = 1.0,
    [int]$ProgressIntervalSec = 15,
    [switch]$SkipAutoFilter,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = "logs/judge_apply_${Split}_${ts}.out.log"
$stderrLog = "logs/judge_apply_${Split}_${ts}.err.log"
$metaLog = "logs/judge_apply_${Split}_${ts}.meta.log"
$arguments = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
  "-Split", $Split,
  "-JudgeModel", $JudgeModel,
  "-JudgeReasoningEffort", $JudgeReasoningEffort,
  "-GeneratorModel", $GeneratorModel,
  "-RequestsPerSecond", "$RequestsPerSecond",
  "-ProgressIntervalSec", "$ProgressIntervalSec"
)
if ($SkipAutoFilter) {
  $arguments += "-SkipAutoFilter"
}
if ($NoResume) {
  $arguments += "-NoResume"
}

$proc = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList $arguments `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -WindowStyle Hidden `
  -PassThru

@(
    "started_at=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
    "pid=$($proc.Id)",
    "split=$Split",
    "run_script=$RunScript",
    "judge_model=$JudgeModel",
    "judge_reasoning_effort=$JudgeReasoningEffort",
    "generator_model=$GeneratorModel",
    "requests_per_second=$RequestsPerSecond",
    "progress_interval_sec=$ProgressIntervalSec",
    "skip_auto_filter=$SkipAutoFilter",
    "resume=$(-not $NoResume)",
    "stdout_log=$stdoutLog",
    "stderr_log=$stderrLog"
) | Set-Content -Encoding UTF8 $metaLog

Write-Host ("Started background process. PID={0}" -f $proc.Id)
Write-Host ("SPLIT : {0}" -f $Split)
Write-Host ("STDOUT: {0}" -f $stdoutLog)
Write-Host ("STDERR: {0}" -f $stderrLog)
Write-Host ("META  : {0}" -f $metaLog)
