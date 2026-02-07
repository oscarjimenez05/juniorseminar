param(
    [switch]$BigCrush = $false
)

$SEED = 2366022249
$IMAGE_NAME = "marzopa/prng-bench"
$RESULTS_DIR = "${PWD}/results"

if (!(Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR }

function Launch-Test ($name, $algo, $delta) {
    $name = $name + "_" + $SEED
    if ($BigCrush){
        $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo | ./test_from_pipe_BigCrush > $reportFile"
        $reportFile = "/app/results/${name}_BigCrush.txt"
    }
    else{
        $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo | ./test_from_pipe > $reportFile"
        $reportFile = "/app/results/${name}_Crush.txt"
    }

    Write-Host "Launching $name... (Report will save to results/${name}_report.txt)"

    $containerName = "${name}_run"

    docker rm -f $containerName 2>$null | Out-Null
    docker run --rm --name $containerName `
        -v "${PWD}/results:/app/results" `
        $IMAGE_NAME `
        sh -c $cmd
}

Write-Host "--- Starting Parallel Tests ---" -ForegroundColor Cyan

Launch-Test "LCG_d13" "lcg" 13
Launch-Test "XOR_d13"  "xor" 13

Write-Host "Done! Check results folder" -ForegroundColor Yellow