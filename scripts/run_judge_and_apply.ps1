param(
    [string]$PythonExe = ".\venv\Scripts\python.exe",
    [string]$Metadata = "data/aug/train_aug_auto_pass_v1.jsonl",
    [string]$JudgeOutput = "data/judge/train_judge_v1.jsonl",
    [string]$AppliedOutput = "data/metadata/train_aug_judge_pass_v1.jsonl",
    [string]$JudgeModel = "gpt-5.5",
    [string]$JudgeReasoningEffort = "low",
    [string]$GeneratorModel = "gpt-5.4-mini",
    [double]$RequestsPerSecond = 1.0,
    [int]$ProgressIntervalSec = 15,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [ValidateSet("INFO", "WARN", "ERROR")]
        [string]$Level,
        [string]$Message
    )
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host ("[{0}] [{1}] {2}" -f $ts, $Level, $Message)
}

function Get-LineCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $JudgeOutput) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $AppliedOutput) | Out-Null

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $Metadata)) {
    throw "Metadata input not found: $Metadata"
}

$runStartedAt = Get-Date
$runId = $runStartedAt.ToString("yyyyMMdd_HHmmss")
$judgeStdout = "logs/judge_llm_${runId}.out.log"
$judgeStderr = "logs/judge_llm_${runId}.err.log"
$applyStdout = "logs/apply_judge_${runId}.out.log"
$applyStderr = "logs/apply_judge_${runId}.err.log"

$inputLines = Get-LineCount -Path $Metadata
Write-Log INFO ("Run started. run_id={0}" -f $runId)
Write-Log INFO ("Inputs: metadata={0}, input_lines={1}, judge_model={2}, judge_reasoning_effort={3}, generator_model={4}, rps={5}, resume={6}" -f $Metadata, $inputLines, $JudgeModel, $JudgeReasoningEffort, $GeneratorModel, $RequestsPerSecond, (-not $NoResume))

try {
    Write-Log INFO "[1/2] Running judge_llm"
    if ((Test-Path $JudgeOutput) -and $NoResume) {
        Remove-Item -LiteralPath $JudgeOutput -Force
        Write-Log WARN ("Existing judge output removed because NoResume was set: {0}" -f $JudgeOutput)
    }

    $judgeStarted = Get-Date
    $judgeArgs = @(
        "-m", "term_ai.augmentation.judge_llm",
        "--metadata", $Metadata,
        "--output", $JudgeOutput,
        "--model", $JudgeModel,
        "--generator-model", $GeneratorModel,
        "--requests-per-second", "$RequestsPerSecond"
    )
    if ($JudgeReasoningEffort) {
        $judgeArgs += @("--reasoning-effort", $JudgeReasoningEffort)
    }
    if (-not $NoResume) {
        $judgeArgs += "--resume"
    }

    $judgeProc = Start-Process `
      -FilePath $PythonExe `
      -ArgumentList $judgeArgs `
      -RedirectStandardOutput $judgeStdout `
      -RedirectStandardError $judgeStderr `
      -WindowStyle Hidden `
      -PassThru

    Write-Log INFO ("judge_llm pid={0}, stdout={1}, stderr={2}" -f $judgeProc.Id, $judgeStdout, $judgeStderr)

    while (-not $judgeProc.HasExited) {
        Start-Sleep -Seconds $ProgressIntervalSec
        $written = Get-LineCount -Path $JudgeOutput
        if ($inputLines -gt 0) {
            $pct = [math]::Round(($written / $inputLines) * 100, 1)
            Write-Log INFO ("judge progress: {0}/{1} rows ({2}%)" -f $written, $inputLines, $pct)
        } else {
            Write-Log INFO ("judge progress: {0} rows written" -f $written)
        }
        $judgeProc.Refresh()
    }
    $judgeProc.Refresh()

    $judgeElapsed = (Get-Date) - $judgeStarted
    $judgeLines = Get-LineCount -Path $JudgeOutput
    Write-Log INFO ("judge_llm finished. exit_code={0}, output_lines={1}, elapsed={2}" -f $judgeProc.ExitCode, $judgeLines, $judgeElapsed)
    if ($judgeProc.ExitCode -ne 0) {
        throw "judge_llm failed with exit code $($judgeProc.ExitCode). stderr_log=$judgeStderr"
    }

    Write-Log INFO "[2/2] Applying judge result"
    $applyStarted = Get-Date
    $applyProc = Start-Process `
      -FilePath $PythonExe `
      -ArgumentList @(
        "-m", "term_ai.augmentation.pipeline", "apply-judge",
        "--metadata", $Metadata,
        "--judge", $JudgeOutput,
        "--output", $AppliedOutput
      ) `
      -RedirectStandardOutput $applyStdout `
      -RedirectStandardError $applyStderr `
      -WindowStyle Hidden `
      -PassThru

    Write-Log INFO ("apply-judge pid={0}, stdout={1}, stderr={2}" -f $applyProc.Id, $applyStdout, $applyStderr)
    Wait-Process -Id $applyProc.Id
    $applyProc.Refresh()
    $applyElapsed = (Get-Date) - $applyStarted
    $appliedLines = Get-LineCount -Path $AppliedOutput
    Write-Log INFO ("apply-judge finished. exit_code={0}, output_lines={1}, elapsed={2}" -f $applyProc.ExitCode, $appliedLines, $applyElapsed)
    if ($applyProc.ExitCode -ne 0) {
        throw "apply-judge failed with exit code $($applyProc.ExitCode). stderr_log=$applyStderr"
    }

    $totalElapsed = (Get-Date) - $runStartedAt
    Write-Log INFO ("Run completed successfully. total_elapsed={0}, judge_lines={1}, applied_lines={2}" -f $totalElapsed, $judgeLines, $appliedLines)
}
catch {
    $totalElapsed = (Get-Date) - $runStartedAt
    Write-Log ERROR ("Run failed after {0}: {1}" -f $totalElapsed, $_.Exception.Message)
    throw
}
