import runpod
import torch
import math
import gc
import base64
import io
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

# Global pipeline
pipe = None

def flush():
    """GPU memory təmizlə"""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except:
        pass

def load_model():
    """Model-i və LoRA-ları yüklə"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    print("Loading model...")
    flush()
    
    # bfloat16 support check
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")
    
    # Scheduler konfiqurasiyası
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Pipeline yüklə
    print("Loading base pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", 
        scheduler=scheduler, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        variant=None, 
        use_safetensors=True
    )
    
    # LoRA-ları yüklə
    print("Loading LoRAs...")
    pipe.load_lora_weights(
        "tarn59/apply_texture_qwen_image_edit_2509", 
        weight_name="apply_texture_v2_qwen_image_edit_2509.safetensors", 
        adapter_name="texture"
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors", 
        adapter_name="lightning"
    )
    
    # Adapter-ləri birləşdir
    print("Fusing LoRAs...")
    pipe.set_adapters(["texture", "lightning"], adapter_weights=[1., 1.])
    pipe.fuse_lora(adapter_names=["texture", "lightning"], lora_scale=1)
    pipe.unload_lora_weights()
    
    # Transformer class dəyişdir
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    
    # Optimizasiyalar
    print("Applying optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    flush()
    print("✅ Model ready!")
    return pipe

def calculate_dimensions(image, max_dim=1024):
    """Şəkil ölçülərini hesabla (32-yə bölünən)"""
    if image is None:
        return 512, 512
    
    original_width, original_height = image.size
    scale = min(max_dim / original_width, max_dim / original_height)
    
    if scale < 1:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        new_width = original_width
        new_height = original_height
    
    # 32-yə bölünən etmək
    new_width = (new_width // 32) * 32
    new_height = (new_height // 32) * 32
    
    # Minimum ölçü
    new_width = max(256, new_width)
    new_height = max(256, new_height)
    
    return new_width, new_height

def decode_base64_image(base64_string):
    """Base64 stringdən PIL Image-ə çevir"""
    try:
        # Data URL prefix sil əgər varsa
        if ',' in base64_string and base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def encode_image_to_base64(image):
    """PIL Image-i base64 stringə çevir"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

def handler(job):
    """
    RunPod Serverless Handler funksiyası
    
    Expected input format:
    {
        "input": {
            "content_image": "base64_encoded_content_image",
            "texture_image": "base64_encoded_texture_image",
            "prompt": "apply wooden texture",
            "seed": 42,                        # optional, default: 42
            "randomize_seed": false,           # optional
            "true_guidance_scale": 1.0,        # optional, default: 1.0
            "num_inference_steps": 4,          # optional, default: 4
            "max_dimension": 1024              # optional, default: 1024
        }
    }
    
    Returns:
    {
        "success": true,
        "image": "base64_encoded_result_image",
        "seed": used_seed_value,
        "width": output_width,
        "height": output_height
    }
    """
    try:
        job_input = job.get("input", {})
        
        # Input validation
        required_fields = ['content_image', 'texture_image', 'prompt']
        for field in required_fields:
            if field not in job_input:
                return {"error": f"Missing required field: '{field}'"}
        
        print("="*60)
        print("🚀 Starting texture application job")
        print("="*60)
        
        # Load model
        print("Loading model...")
        model = load_model()
        
        # Decode images
        print("Decoding images...")
        content_image = decode_base64_image(job_input['content_image'])
        texture_image = decode_base64_image(job_input['texture_image'])
        prompt = job_input['prompt']
        
        # Parameters
        seed = job_input.get('seed', 42)
        randomize_seed = job_input.get('randomize_seed', False)
        true_guidance_scale = job_input.get('true_guidance_scale', 1.0)
        num_inference_steps = job_input.get('num_inference_steps', 4)
        max_dimension = job_input.get('max_dimension', 1024)
        
        print(f"📝 Prompt: {prompt}")
        print(f"🎲 Seed: {seed} (randomize: {randomize_seed})")
        print(f"⚙️  Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")
        
        # Flush before generation
        flush()
        
        # Randomize seed if requested
        if randomize_seed:
            import random
            seed = random.randint(0, 2147483647)
            print(f"🎲 New random seed: {seed}")
        
        # Generator
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Calculate dimensions
        width, height = calculate_dimensions(content_image, max_dim=max_dimension)
        print(f"📐 Output dimensions: {width}x{height}")
        
        # Prepare images
        content_pil = content_image.convert("RGB")
        texture_pil = texture_image.convert("RGB")
        pil_images = [content_pil, texture_pil]
        
        print("🎨 Generating texture application...")
        
        # Generate with torch.no_grad for memory efficiency
        with torch.no_grad():
            result = model(
                image=pil_images,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_guidance_scale,
                num_images_per_prompt=1,
            ).images[0]
        
        # Flush after generation
        flush()
        
        print("✅ Generation completed!")
        
        # Encode result
        print("📦 Encoding result...")
        result_base64 = encode_image_to_base64(result)
        
        print("="*60)
        print("✅ Job completed successfully!")
        print("="*60)
        
        return {
            "success": True,
            "image": result_base64,
            "seed": seed,
            "width": width,
            "height": height,
            "prompt": prompt
        }
        
    except Exception as e:
        # Cleanup on error
        flush()
        
        print("="*60)
        print(f"❌ Error: {str(e)}")
        print("="*60)
        
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }

# Pre-load model on container start
print("🚀 RunPod Serverless Worker Starting...")
print("="*60)

try:
    print("Pre-loading model...")
    load_model()
    print("✅ Model pre-loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not pre-load model: {e}")
    print("Model will be loaded on first request")

print("="*60)
print("🎯 Worker ready! Waiting for requests...")
print("="*60)

# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})