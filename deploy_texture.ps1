$DOCKER_USERNAME = "rufatshikhiyev"  
$IMAGE_NAME = "naringi-texture-serverless"
$VERSION = "v1.0.0"

$FULL_IMAGE_NAME = "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
$LATEST_TAG = "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

Write-Host "========================================"
Write-Host "Naringi-AI Deployment Starting"
Write-Host "========================================"
Write-Host "Image: $FULL_IMAGE_NAME"
Write-Host ""

# ========== STEP 1: Docker Yoxla ==========
Write-Host "[1/5] Docker Desktop yoxlanir..."

try {
    $dockerVersion = docker --version
    Write-Host "Docker tapildi: $dockerVersion"
    Write-Host ""
} catch {
    Write-Host "Docker Desktop quraşdirilmayib!"
    Write-Host "Quraşdirin: https://www.docker.com/products/docker-desktop"
    exit 1
}

# ========== STEP 2: Faylları Yoxla ==========
Write-Host "[2/5] Lazimi fayllar yoxlanir..."

$requiredFiles = @(
    "Dockerfile_texture",
    "rp_handler_texture.py",
    "qwenimage",
    "requirements.txt"
)

foreach ($file in $requiredFiles) {
    if (-Not (Test-Path $file)) {
        Write-Host "Fayl tapilmadi: $file"
        exit 1
    }
}
Write-Host "Butun fayllar movcuddur"
Write-Host ""

# ========== STEP 3: Docker Login ==========
Write-Host "[3/5] Docker Hub login..."
docker login
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker login ugursuz!"
    exit 1
}
Write-Host "Login ugurlu"
Write-Host ""

# ========== STEP 4: Handler Hazirla ==========
Write-Host "[4/5] Handler hazirllanir..."
Copy-Item -Path "rp_handler_texture.py" -Destination "rp_handler.py" -Force
Write-Host "Handler kopyalandi"
Write-Host ""

# ========== STEP 5: Docker Build ==========
Write-Host "[5/5] Docker image build edilir..."
Write-Host "Bu 10-20 deqiqe ceke biler..."
Write-Host ""

docker buildx build --platform linux/amd64 -f Dockerfile_texture -t $FULL_IMAGE_NAME -t $LATEST_TAG --progress=plain .

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Build ugursuz!"
    Remove-Item "rp_handler.py" -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host "Build ugurlu!"
Write-Host ""

# ========== STEP 6: Docker Push ==========
Write-Host "[6/6] Docker Hub-a push edilir..."

docker push $FULL_IMAGE_NAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push ugursuz!"
    exit 1
}

docker push $LATEST_TAG
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push (latest) ugursuz!"
    exit 1
}

# Cleanup
Remove-Item "rp_handler.py" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================"
Write-Host "DEPLOYMENT TAMAMLANDI!"
Write-Host "========================================"
Write-Host ""
Write-Host "Image: $FULL_IMAGE_NAME"
Write-Host ""
Write-Host "Novbeti addimlar:"
Write-Host "1. https://console.runpod.io/serverless"
Write-Host "2. New Endpoint"
Write-Host "3. Docker Image: $FULL_IMAGE_NAME"
Write-Host "4. GPU: 24GB+ (A5000/A6000/RTX4090)"
Write-Host "5. Deploy"
Write-Host ""