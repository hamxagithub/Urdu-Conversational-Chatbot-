import torch

print("=== Checking actual model state dict ===")
try:
    checkpoint = torch.load('files/best_model.pth', map_location='cpu')
    model_state = checkpoint['model_state_dict']
    print(f"Model state dict type: {type(model_state)}")
    print(f"Number of parameters: {len(model_state)}")
    print("\nFirst 20 parameter keys:")
    for i, k in enumerate(list(model_state.keys())):
        print(f"  {i+1}: {k}")
        if i >= 19:  # Show first 20
            break
    
    print(f"\n... and {len(model_state.keys())-20} more keys" if len(model_state.keys()) > 20 else "")
    
    # Check for specific patterns
    encoder_keys = [k for k in model_state.keys() if 'encoder' in k]
    decoder_keys = [k for k in model_state.keys() if 'decoder' in k]
    print(f"\nEncoder keys found: {len(encoder_keys)}")
    print(f"Decoder keys found: {len(decoder_keys)}")
    
    if encoder_keys:
        print("Sample encoder keys:")
        for k in encoder_keys[:5]:
            print(f"  {k}")
            
except Exception as e:
    print(f"Error: {e}")
