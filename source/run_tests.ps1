$SEED = 12345
$IMAGE_NAME = "prng-bench"
$RESULTS_DIR = "${PWD}/results"

New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

function Launch-Test ($name, $algo, $delta) {
    Write-Host "Launching $name..." -NoNewline

    $logFile = "/app/results/${name}_progress.log"

    $reportFile = "/app/results/${name}_report.txt"

    $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo 2> $logFile | ./test_from_pipe > $reportFile"

    $id = docker run -d --rm -v "${RESULTS_DIR}:/app/results" $IMAGE_NAME sh -c $cmd

    Write-Host " [OK] Container ID: $($id.Substring(0,8))"
}

Write-Host "--- Starting Parallel Tests (Seed: $SEED) ---" -ForegroundColor Cyan

# 1. LCG Non-Overlapping
Launch-Test "LCG_Noverlap" "lcg" 0

# 2. LCG Delta=13
Launch-Test "LCG_Delta13"  "lcg" 13

# 3. XOR Non-Overlapping
Launch-Test "XOR_Noverlap" "xor" 0

# 4. XOR Delta=13
Launch-Test "XOR_Delta13"  "xor" 13

Write-Host "Done! Check results folder" -ForegroundColor Yellow