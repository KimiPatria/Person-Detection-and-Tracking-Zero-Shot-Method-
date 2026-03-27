import torch
from models.factory import build_net

# 1. Create a Wrapper to export the exact testing function
class DAINet_ONNX_Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, x):
        # We specifically export 'test_forward' since that is what test.py uses
        return self.net.test_forward(x)[0]

if __name__ == '__main__':
    print("Loading PyTorch model...")
    # 2. Load the network and your trained weights
    net = build_net('test', num_classes=2, model='dark')
    
    # Load to CPU for a safer export process
    net.load_state_dict(torch.load('weights/dsfd.pth', map_location='cpu'))
    net.eval()
    
    # Wrap the model
    wrapped_net = DAINet_ONNX_Wrapper(net)
    wrapped_net.eval()

    # 3. Create a "Dummy Input" 
    # This shows ONNX the expected shape (Batch=1, Channels=3, Height=640, Width=640)
    dummy_input = torch.randn(1, 3, 640, 640)

    print("Exporting to ONNX (This may take a minute)...")
    # 4. Export the model
    torch.onnx.export(
        wrapped_net,                 # Model being run
        dummy_input,                 # Model input
        "weights/dsfd_optimized.onnx", # Where to save the file
        export_params=True,          # Store the trained weights inside the ONNX file
        opset_version=11,            # Version 11 is highly stable for Object Detection
        do_constant_folding=True,    # Automatically optimize and speed up mathematical layers
        input_names=['input'],       # Name the input channel
        output_names=['output'],     # Name the output channel
        dynamic_axes={               # Allow the model to accept ANY image resolution later
            'input': {2: 'height', 3: 'width'},
            'output': {0: 'batch'}
        }
    )
    print("Success! Optimized model saved to weights/dsfd_optimized.onnx")