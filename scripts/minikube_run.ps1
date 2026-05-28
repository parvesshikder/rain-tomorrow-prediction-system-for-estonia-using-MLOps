param(
    [int]$Cpus = 4,
    [int]$MemoryMb = 4096
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Starting Minikube..."
minikube start --driver=docker --cpus=$Cpus --memory=$MemoryMb
Assert-LastExitCode "minikube start"

Write-Host "Building image inside Minikube..."
minikube image build -t estonia-rain-mlops:latest .
Assert-LastExitCode "minikube image build"

Write-Host "Deploying API manifests..."
kubectl apply -f k8s/namespace.yaml
Assert-LastExitCode "kubectl apply namespace"
kubectl apply -f k8s/api-deployment.yaml
Assert-LastExitCode "kubectl apply deployment"
kubectl apply -f k8s/api-service.yaml
Assert-LastExitCode "kubectl apply service"

Write-Host "Fetching service URL..."
$Url = minikube service -n rain-mlops rain-api --url
Assert-LastExitCode "minikube service url"
Write-Host "API URL: $Url"
