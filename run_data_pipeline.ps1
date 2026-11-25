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

$outputPath = Join-Path $PSScriptRoot "breath_stats.txt"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# Clear the report before starting using UTF-8 without BOM.
[System.IO.File]::WriteAllText($outputPath, "", $utf8NoBom)

try {
    foreach ($file in $files) {
        Write-Host "`nProcessing: $($file.Name)"

        $pythonOutput = & python $wrapperPath $file.Name

        foreach ($line in $pythonOutput) {
            Write-Host $line
            [System.IO.File]::AppendAllText(
                $outputPath,
                $line + [Environment]::NewLine,
                $utf8NoBom
            )
        }

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