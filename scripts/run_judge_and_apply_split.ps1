param(
    [ValidateSet("train", "dev", "test")]
    [string]$Split = "train",
    [string]$PythonExe = ".\venv\Scripts\python.exe",
    [string]$JudgeModel = "gpt-5.5",
    [string]$JudgeReasoningEffort = "low",
    [string]$GeneratorModel = "gpt-5.4-mini",
    [double]$RequestsPerSecond = 1.0,
    [int]$ProgressIntervalSec = 15,
    [switch]$SkipAutoFilter,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

$candidateMetadata = "data/aug/${Split}_aug_candidate_v1.jsonl"
$metadata = "data/aug/${Split}_aug_auto_pass_v1.jsonl"
$judgeOutput = "data/judge/${Split}_judge_v1.jsonl"
$appliedOutput = "data/metadata/${Split}_aug_judge_pass_v1.jsonl"

if (-not $SkipAutoFilter) {
    if (-not (Test-Path $candidateMetadata)) {
        throw "Candidate metadata not found: $candidateMetadata"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $metadata) | Out-Null
    Write-Host ("[0/2] Auto-filtering {0} candidate metadata..." -f $Split)
    & $PythonExe -m term_ai.augmentation.pipeline auto-filter `
      --metadata $candidateMetadata `
      --output $metadata
    if ($LASTEXITCODE -ne 0) {
        throw "auto-filter failed with exit code $LASTEXITCODE"
    }
}

& ".\scripts\run_judge_and_apply.ps1" `
  -PythonExe $PythonExe `
  -Metadata $metadata `
  -JudgeOutput $judgeOutput `
  -AppliedOutput $appliedOutput `
  -JudgeModel $JudgeModel `
  -JudgeReasoningEffort $JudgeReasoningEffort `
  -GeneratorModel $GeneratorModel `
  -RequestsPerSecond $RequestsPerSecond `
  -ProgressIntervalSec $ProgressIntervalSec `
  -NoResume:$NoResume
