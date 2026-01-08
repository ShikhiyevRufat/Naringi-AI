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
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except:
        pass

def load_model():
    """Model-i yüklə"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    print("Loading model...")
    flush()
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
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
    
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", 
        scheduler=scheduler, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        variant=None, 
        use_safetensors=True
    )
    
    print("Loading LoRAs...")
    pipe.load_lora_weights("tarn59/apply_texture_qwen_image_edit_2509", 
            weight_name="apply_texture_v2_qwen_image_edit_2509.safetensors", adapter_name="texture")
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors", adapter_name="lightning")
    
    pipe.set_adapters(["texture", "lightning"], adapter_weights=[1., 1.])
    pipe.fuse_lora(adapter_names=["texture", "lightning"], lora_scale=1)
    pipe.unload_lora_weights()
    
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    
    print("Applying optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    flush()
    print("Model ready!")
    return pipe

def calculate_dimensions(image, max_dim=1024):
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
    
    new_width = (new_width // 32) * 32
    new_height = (new_height // 32) * 32
    
    new_width = max(256, new_width)
    new_height = max(256, new_height)
    
    return new_width, new_height

def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job["input"]
        
        # Validate inputs
        if not job_input.get('content_image'):
            return {"error": "content_image is required"}
        if not job_input.get('texture_image'):
            return {"error": "texture_image is required"}
        if not job_input.get('prompt'):
            return {"error": "prompt is required"}
        
        # Load model
        model = load_model()
        
        # Decode images
        content_image = decode_base64_image(job_input['content_image'])
        texture_image = decode_base64_image(job_input['texture_image'])
        prompt = job_input['prompt']
        
        # Parameters
        seed = job_input.get('seed', 42)
        randomize_seed = job_input.get('randomize_seed', False)
        true_guidance_scale = job_input.get('true_guidance_scale', 1.0)
        num_inference_steps = job_input.get('num_inference_steps', 4)
        max_dimension = job_input.get('max_dimension', 1024)
        
        flush()
        
        if randomize_seed:
            import random
            seed = random.randint(0, 2147483647)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        width, height = calculate_dimensions(content_image, max_dim=max_dimension)
        
        print(f"Generating: {width}x{height}, prompt: {prompt}")
        
        content_pil = content_image.convert("RGB")
        texture_pil = texture_image.convert("RGB")
        pil_images = [content_pil, texture_pil]
        
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
        
        flush()
        
        result_base64 = encode_image_to_base64(result)
        
        return {
            "success": True,
            "image": result_base64,
            "seed": seed,
            "width": width,
            "height": height
        }
        
    except Exception as e:
        flush()
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# RunPod serverless başlat
runpod.serverless.start({"handler": handler})