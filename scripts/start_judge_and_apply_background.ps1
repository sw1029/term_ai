param(
    [string]$RunScript = ".\scripts\run_judge_and_apply.ps1",
    [string]$JudgeModel = "gpt-5.5",
    [string]$JudgeReasoningEffort = "low",
    [string]$GeneratorModel = "gpt-5.4-mini",
    [double]$RequestsPerSecond = 1.0,
    [int]$ProgressIntervalSec = 15,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = "logs/judge_apply_${ts}.out.log"
$stderrLog = "logs/judge_apply_${ts}.err.log"
$metaLog = "logs/judge_apply_${ts}.meta.log"
$arguments = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
  "-JudgeModel", $JudgeModel,
  "-JudgeReasoningEffort", $JudgeReasoningEffort,
  "-GeneratorModel", $GeneratorModel,
  "-RequestsPerSecond", "$RequestsPerSecond",
  "-ProgressIntervalSec", "$ProgressIntervalSec"
)
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
    "run_script=$RunScript",
    "judge_model=$JudgeModel",
    "judge_reasoning_effort=$JudgeReasoningEffort",
    "generator_model=$GeneratorModel",
    "requests_per_second=$RequestsPerSecond",
    "progress_interval_sec=$ProgressIntervalSec",
    "resume=$(-not $NoResume)",
    "stdout_log=$stdoutLog",
    "stderr_log=$stderrLog"
) | Set-Content -Encoding UTF8 $metaLog

Write-Host ("Started background process. PID={0}" -f $proc.Id)
Write-Host ("STDOUT: {0}" -f $stdoutLog)
Write-Host ("STDERR: {0}" -f $stderrLog)
Write-Host ("META  : {0}" -f $metaLog)
