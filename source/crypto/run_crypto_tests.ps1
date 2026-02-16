param(
    [switch]$BigCrush = $false
)

$SEED = 2366022249
$IMAGE_NAME = "marzopa/prng-bench"

$RESULTS_DIR = "${PWD}/results"
if (!(Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR }

function Launch-Test ($name, $delta) {
    $fullName = "${name}_${SEED}"

    if ($BigCrush){
        $reportFile = "crypto/results/${fullName}_BigCrush.txt"
        $binary = "./test_from_pipe_BigCrush"
    }
    else{
        $reportFile = "crypto/results/${fullName}_Crush.txt"
        $binary = "./test_from_pipe"
    }

    $cmd = "python3 crypto/crypto_testing_interface.py $SEED --delta $delta | $binary > $reportFile"

    Write-Host "Launching $fullName... (Output: $RESULTS_DIR\${fullName}_...)"

    $containerName = "${fullName}_run"

    docker rm -f $containerName 2>$null | Out-Null

    docker run -d --rm --name $containerName `
        -v "${RESULTS_DIR}:/app/crypto/results" `
        $IMAGE_NAME `
        sh -c $cmd
}

Write-Host "--- Starting CryptoLehmer TestU01 Tests ---" -ForegroundColor Cyan

Launch-Test "CryptoLehmer_d0" 0

Write-Host "Tests launched! Check the 'results' folder for logs." -ForegroundColor Yellow