param(
    [switch]$BigCrush = $false
)


$SEED = 2366022249
$IMAGE_NAME = "marzopa/prng-bench"
$RESULTS_DIR = "${PWD}/results"

if (!(Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR }

function Launch-Test ($name, $algo, $delta) {
    Write-Host "Launching $name..." -NoNewline

    if ($BigCrush){
        $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo | ./test_from_pipe_BigCrush"
    }
    else{
        $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo | ./test_from_pipe"
    }

    $containerName = "${name}_run"

    docker rm -f $containerName 2>$null | Out-Null

    $id = docker run -d --name $containerName $IMAGE_NAME sh -c $cmd

    # Start a background process to tail the logs into your local file
    Start-Job -ScriptBlock {
        param($cName, $resDir, $tName)
        docker logs -f $cName > "$resDir/${tName}_report.txt" 2> "$resDir/${tName}_progress.log"
    } -ArgumentList $containerName, $RESULTS_DIR, $name | Out-Null

    Write-Host " [OK] Monitoring Seed: $SEED"
}

Write-Host "--- Starting Parallel Tests ---" -ForegroundColor Cyan

Launch-Test "LCG_d13" "lcg" 13
Launch-Test "XOR_d13"  "xor" 13

Write-Host "Done! Check results folder" -ForegroundColor Yellow