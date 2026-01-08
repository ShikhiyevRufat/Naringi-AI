mkdir builder
cat > builder/setup.sh << 'EOF'
#!/bin/bash

# Əlavə paketlər quraşdır
echo "Installing additional packages..."

# Əgər Qwen package pip-də yoxdursa
pip install git+https://github.com/QwenLM/Qwen-Image-Edit.git

# Model-ləri cache-lə (optional - ilk işə salma sürətlənir)
# python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-Edit-2509')"

echo "Setup complete!"
EOF

chmod +x builder/setup.sh