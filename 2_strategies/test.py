# Test Pytorch is using Cuda Tensors for GPU Processing
import torch

# x = torch.rand(5, 3)
# print(x)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(device)

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())




# print(torch.cuda.is_available()) 
# # Expected Output = True

# print(torch.cuda.device_count()) 
# # Expected Output = 1

# print(torch.cuda.current_device()) 
# # Expected Output = 0

# print(torch.cuda.device(0)) 
# # Expected Output = instance_name

# print(torch.cuda.get_device_name()) 
# # Expected = 2070 rtx
