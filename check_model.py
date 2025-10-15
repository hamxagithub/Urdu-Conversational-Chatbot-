import torch
import pickle

print("=== Checking best_model.pth ===")
try:
    state = torch.load('files/best_model.pth', map_location='cpu')
    print(f"Type: {type(state)}")
    if isinstance(state, dict):
        print("Keys in state_dict:")
        for i, k in enumerate(list(state.keys())):
            print(f"  {i+1}: {k}")
            if i > 30:  # Limit output
                print(f"  ... and {len(state.keys())-31} more keys")
                break
    else:
        print("State is not a dictionary")
except Exception as e:
    print(f"Error loading .pth: {e}")

print("\n=== Checking best_model.pkl ===")
try:
    with open('files/best_model.pkl', 'rb') as f:
        state = pickle.load(f)
    print(f"Type: {type(state)}")
    if isinstance(state, dict):
        print("Keys in state_dict:")
        for i, k in enumerate(list(state.keys())):
            print(f"  {i+1}: {k}")
            if i > 30:  # Limit output
                print(f"  ... and {len(state.keys())-31} more keys")
                break
    else:
        print("State is not a dictionary")
except Exception as e:
    print(f"Error loading .pkl: {e}")