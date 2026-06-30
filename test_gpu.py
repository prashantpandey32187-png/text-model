"""
Test GPU availability and acceleration options
"""
import torch
import sys
import platform

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 60)
print("GPU Detection Test")
print("=" * 60)

# Check PyTorch CUDA
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Check for NVIDIA GPU via CUDA
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"\n✅ NVIDIA GPU(s) Detected: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
    
    # Test CUDA with a simple operation
    print("\n[TEST] Running CUDA test...")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"   ✅ CUDA test successful! Result shape: {z.shape}")
    print(f"   Device: {z.device}")
    
    # Test cuDNN
    print("\n[TEST] Testing cuDNN...")
    torch.backends.cudnn.enabled = True
    print(f"   ✅ cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
else:
    print("\n[INFO] No NVIDIA GPU detected via CUDA.")
    
    # Check for DirectML (AMD GPU support)
    try:
        import onnxruntime as ort
        # Check available providers (different API for onnxruntime-directml)
        try:
            providers = ort.get_available_providers()
        except AttributeError:
            providers = ort.capi._pybind_state.get_available_providers()
        print(f"\n🔍 ONNX Runtime providers: {providers}")
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML Execution Provider available!")
            print("   Your AMD GPU can be used via ONNX Runtime DirectML")
            print("   This provides GPU acceleration for AI models on AMD GPUs")
            
            # Get DML device info
            import subprocess
            try:
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-Table -AutoSize'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print(f"\n📋 Available GPUs:")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(f"   {line}")
            except:
                pass
        else:
            print("❌ DirectML Execution Provider NOT available")
    except ImportError:
        print("   ONNX Runtime not installed for DirectML checks")
    
    # Try to get GPU info via system
    print("\n[INFO] System Graphics:")
    try:
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command', 
             'Get-CimInstance -ClassName Win32_VideoController | Select-Object Name | Format-Table -HideTableHeaders'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")
    except:
        print("   Unable to query GPU information")

print("\n" + "=" * 60)