# import numpy as np

# def view_npy_file(file_path, num_rows=1000):
#     """Load an .npy file and display its contents properly."""
#     try:
#         data = np.load(file_path, allow_pickle=True)

#         # Handle a 0D object array (likely a dictionary)
#         if data.shape == ():  # Empty tuple means scalar
#             print(f"📌 Detected 0D NumPy Array - Extracting its contents...")
#             data = data.item()  # Extract stored object (dictionary or list)

#         print(f"📂 Loaded .npy file: {file_path}")
#         print(f"🔹 Data Type: {type(data)}")

#         # If it's a dictionary, print keys and values
#         if isinstance(data, dict):
#             print(f"📌 Dictionary detected! {len(data.keys())} keys found.")
#             keys = list(data.keys())[:num_rows]
#             for key in keys:
#                 value = data[key]
#                 print(f"🔑 Key: {key} | Type: {type(value)} | Shape: {getattr(value, 'shape', 'Scalar')}")
        
#         # If it's an array, print shape and preview
#         elif isinstance(data, np.ndarray):
#             print(f"📌 NumPy Array detected! Shape: {data.shape}, Dtype: {data.dtype}")
#             print(data[:num_rows] if data.ndim > 0 else data)

#         else:
#             print(f"📌 Unknown Format: Printing raw content:")
#             print(data)

#     except Exception as e:
#         print(f"❌ Error loading {file_path}: {e}")

# # Example usage
# # view_npy_file("../data/visual_feature.npy")
# view_npy_file("../data/training_dict.npy")



import numpy as np

def view_npy_file(file_path, num_keys=10):
    """Load an .npy file and display its structure and sample data intelligently."""
    try:
        data = np.load(file_path, allow_pickle=True)

        # Case 1: If it's a 0D NumPy array (likely a stored dictionary), extract the object
        if data.shape == ():  
            print(f"📌 Detected 0D NumPy Array - Extracting its contents...")
            data = data.item()  # Extract the stored object

        print(f"📂 Loaded .npy file: {file_path}")
        print(f"🔹 Data Type: {type(data)}")

        # Case 2: If it's a dictionary
        if isinstance(data, dict):
            print(f"📌 Dictionary detected! {len(data.keys())} keys found.")
            print("-" * 50)
            
            # Print first `num_keys` entries
            for idx, (key, value) in enumerate(data.items()):
                if isinstance(value, np.ndarray):
                    value_type = f"NumPy Array | Shape: {value.shape} | Dtype: {value.dtype}"
                    sample_value = value[:10] if value.ndim > 0 else value
                elif isinstance(value, list):
                    value_type = f"List | Length: {len(value)}"
                    sample_value = value[:20]
                else:
                    value_type = f"{type(value).__name__}"
                    sample_value = value
                
                print(f"🔑 Key: {key} | {value_type} | Sample: {sample_value}")
                print ("----------------")
                # print (data[key])
                if idx + 1 >= num_keys:
                    break

        # Case 3: If it's a NumPy array
        elif isinstance(data, np.ndarray):
            print(f"📌 NumPy Array detected! Shape: {data.shape}, Dtype: {data.dtype}")
            print(data[:num_keys] if data.ndim > 0 else data)

        else:
            print(f"📌 Unknown Format: {type(data)}")
            print(data)

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")

# Example usage
print("\n🔍 **Visual Feature File:**")
view_npy_file("../data/visual_feature.npy")

print("\n🔍 **Category Feature File:**")
view_npy_file("../data/category_feature.npy")

print("\n🔍 **Training File:**")
view_npy_file("../data/training_dict.npy")


print("\n🔍 **Testing File:**")
view_npy_file("../data/testing_dict.npy")


print("\n🔍 **Validate File:**")
view_npy_file("../data/validation_dict.npy")
