#!/bin/bash

DOCKER_USERNAME="rufatshikhiyev"  
IMAGE_NAME="naringi-texture-serverless"
VERSION="v1.0.0"

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "Naringi-AI Texture Application Deployment"
echo "=========================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo ""


if [ "${DOCKER_USERNAME}" == "naringi" ]; then
    echo "❌ ERROR: Docker username dəyişdirilməyib!"
    echo "deploy.sh faylını açın və DOCKER_USERNAME-i öz username-inizlə dəyişdirin"
    exit 1
fi

# Step 1: Docker login
echo "Step 1: Docker Hub-a login..."
docker login
if [ $? -ne 0 ]; then
    echo "❌ Error: Docker login failed!"
    exit 1
fi

# Step 2: Copy handler
echo ""
echo "Step 2: Handler faylı hazırlanır..."
cp rp_handler_texture.py rp_handler.py

# Step 3: Build Docker image
echo ""
echo "Step 3: Docker image build edilir..."
echo "⚠️  Bu proses 10-20 dəqiqə çəkə bilər (model download olunur)..."

# M1/M2 Mac üçün --platform linux/amd64
# Intel Mac və ya Linux üçün bu flag-i silə bilərsiniz
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile_texture \
    -t ${FULL_IMAGE_NAME} \
    -t ${LATEST_TAG} \
    --progress=plain \
    .

if [ $? -ne 0 ]; then
    echo "❌ Error: Docker build failed!"
    exit 1
fi

echo "✅ Build successful!"

# Step 4: Push to Docker Hub
echo ""
echo "Step 4: Docker Hub-a push edilir..."
echo "⚠️  Bu proses bir neçə dəqiqə çəkə bilər..."

docker push ${FULL_IMAGE_NAME}
if [ $? -ne 0 ]; then
    echo "❌ Error: Docker push failed!"
    exit 1
fi

docker push ${LATEST_TAG}
if [ $? -ne 0 ]; then
    echo "❌ Error: Docker push (latest) failed!"
    exit 1
fi

# Cleanup
rm -f rp_handler.py

echo ""
echo "=========================================="
echo "✅ Deploy uğurla tamamlandı!"
echo "=========================================="
echo ""
echo "🐳 Docker Image: ${FULL_IMAGE_NAME}"
echo ""
echo "📋 Növbəti addımlar:"
echo ""
echo "1️⃣  RunPod Console-a gedin:"
echo "    https://console.runpod.io/serverless"
echo ""
echo "2️⃣  'New Endpoint' düyməsinə klikləyin"
echo ""
echo "3️⃣  Konfiqurasiya:"
echo "    • Custom Source → Docker Image seçin"
echo "    • Image: ${FULL_IMAGE_NAME}"
echo "    • GPU: 24GB+ (A5000, A6000, RTX 4090)"
echo "    • Min Workers: 0 (test üçün 1)"
echo "    • Max Workers: 3-5"
echo "    • Container Disk: 20GB+"
echo "    • Timeout: 300s"
echo ""
echo "4️⃣  'Deploy' düyməsinə klikləyin"
echo ""
echo "5️⃣  Endpoint hazır olduqdan sonra test edin:"
echo ""
echo "Python test:"
echo "  from naringi_texture_client import NaringiTextureClient"
echo "  client = NaringiTextureClient('YOUR_ENDPOINT_ID', 'YOUR_API_KEY')"
echo "  result = client.apply_texture('content.jpg', 'texture.jpg', 'apply texture')"
echo ""
echo "cURL test:"
echo "  curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\"
echo "    -H 'Authorization: Bearer YOUR_API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\": {...}}'"
echo ""
echo "=========================================="
echo "📚 Daha ətraflı: RUNPOD_DEPLOYMENT.md"
echo "=========================================="