import torch
import platform
import subprocess
import sys
import os

def check_gpu_availability():
    """Comprehensive GPU detection and setup"""
    print("🔍 GPU Detection and Setup")
    print("=" * 50)
    
    # System info
    print(f"🖥️ OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print()
    
    # CUDA Check
    print("🎮 CUDA Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Version: {torch.version.cuda if torch.version.cuda else 'Not installed'}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {gpu_props.name}")
            print(f"     Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"     Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        print("   ❌ No CUDA GPUs found")
    print()
    
    # Check NVIDIA drivers
    print("🚗 NVIDIA Driver Check:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ NVIDIA drivers installed")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"   Driver Version: {driver_version}")
                    break
        else:
            print("   ❌ nvidia-smi failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
    print()
    
    # Check environment variables
    print("🌍 Environment Variables:")
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    print(f"   CUDA_PATH: {cuda_path}")
    print(f"   CUDA_HOME: {cuda_home}")
    print()
    
    # MPS Check (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print("🍎 Apple MPS Status:")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   MPS Built: {torch.backends.mps.is_built()}")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    
    if not torch.cuda.is_available():
        print("   🔧 To enable CUDA:")
        print("   1. Install NVIDIA GPU drivers from: https://www.nvidia.com/drivers/")
        print("   2. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("   3. Reinstall PyTorch with CUDA:")
        print("      pip uninstall torch torchvision")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("   🔍 Check your GPU model:")
        print("   - Right-click on Desktop -> Display settings -> Advanced display")
        print("   - Or run: wmic path win32_VideoController get name")
    
    if torch.cuda.is_available():
        print("   ✅ CUDA is ready!")
        print("   🚀 Your system can use GPU acceleration")
    
    return torch.cuda.is_available()

def test_gpu_pytorch():
    """Test PyTorch GPU functionality"""
    print("\n🧪 PyTorch GPU Test:")
    print("-" * 30)
    
    try:
        # Test basic tensor operations
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Create test tensors
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            
            # Time GPU operation
            import time
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start
            
            print(f"🚀 GPU Matrix Multiplication: {gpu_time:.4f} seconds")
            
            # Compare with CPU
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            start = time.time()
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start
            
            print(f"🐌 CPU Matrix Multiplication: {cpu_time:.4f} seconds")
            print(f"⚡ GPU Speedup: {cpu_time/gpu_time:.2f}x faster")
            
        else:
            print("❌ No GPU available for testing")
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

def get_gpu_info():
    """Get detailed GPU information"""
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n🎮 Detected Graphics Cards:")
            print("-" * 30)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                line = line.strip()
                if line and 'Name' not in line:
                    print(f"   {line}")
    except:
        print("❌ Could not detect graphics cards")

if __name__ == "__main__":
    # Run all checks
    get_gpu_info()
    has_gpu = check_gpu_availability()
    
    if has_gpu:
        test_gpu_pytorch()
    
    print("\n" + "="*50)
    if has_gpu:
        print("🎉 Your system is ready for GPU acceleration!")
    else:
        print("💻 Your system will use CPU processing")
    print("="*50)
