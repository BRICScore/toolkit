Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$wrapperPath = Join-Path $PSScriptRoot ".run_pipeline_no_upload.py"

@'
import io
import sys
import data_processing_pipeline as pipeline

class NoUploadHandler:
    def uploadMeasurement(self, filepath_raw, filepath_clean, filepath_features):
        print("Database upload skipped")

pipeline.DatabaseHandler = NoUploadHandler
sys.stdin = io.StringIO(f"{sys.argv[1]}\n1\n1\n")
pipeline.main()
'@ | Set-Content -LiteralPath $wrapperPath -Encoding ascii

$files = Get-ChildItem -LiteralPath "data" -File -Filter "*.jsonl"

if ($files.Count -eq 0) {
    Write-Host "No JSONL files found in toolkit/data."
    Remove-Item -LiteralPath $wrapperPath -Force
    exit 1
}

try {
    foreach ($file in $files) {
        Write-Host "`nProcessing: $($file.Name)"

        python $wrapperPath $file.Name

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed: $($file.Name)"
            continue
        }

        Write-Host "Completed: $($file.Name)"
    }
}
finally {
    Remove-Item -LiteralPath $wrapperPath -Force
}