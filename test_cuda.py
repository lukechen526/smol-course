import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name of the current GPU
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"Current GPU name: {gpu_name}")

    # Allocate a tensor on the GPU
    device = torch.device("cuda")
    tensor = torch.randn(3, 3).to(device)
    print(f"Tensor on GPU: {tensor}")

    # Check if model is on Cuda
    model = torch.nn.Linear(10, 2)
    model.to(device)
    print(f"Model on GPU: {next(model.parameters()).is_cuda}")

    # You can also check the CUDA version
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available. Running on CPU.")
