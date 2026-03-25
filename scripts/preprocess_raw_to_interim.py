import os
import scipy.io
import pandas as pd
import numpy as np
import re

RAW_DATA_PATH = "data/raw"
INTERIM_DATA_PATH = "data/interim"

def get_mat_metadata(filename, category):
    """
    Extracts metadata from filename and category.
    Example: DE_IR007_0.mat -> Fault=IR, Diameter=0.007, Load=0
    """
    # Pattern for fault files: [Label]_[Load].mat
    # Labels like IR007, B014, OR021@6
    
    label_part = filename.replace('.mat', '')
    if 'Normal' in label_part:
        load = label_part.split('_')[1]
        return "Normal", 0.0, int(load), category
    
    # Fault patterns
    match = re.match(r"([A-Z]+)(\d+)(@\d+)?_(\d+)", label_part)
    if match:
        fault_type = match.group(1)
        diameter = int(match.group(2)) / 1000.0
        position = match.group(3) if match.group(3) else ""
        load = int(match.group(4))
        
        full_fault = f"{fault_type}{position}"
        return full_fault, diameter, load, category
    
    return "Unknown", 0.0, 0, category

def preprocess_mat_file(file_path, dest_path, metadata):
    try:
        mat = scipy.io.loadmat(file_path)
        
        # Find variable names (keys usually start with X)
        keys = [k for k in mat.keys() if k.startswith('X') and '_DE_time' in k]
        if not keys:
            # Fallback for some files that might use different keys
            keys = [k for k in mat.keys() if '_DE_time' in k]
        
        if not keys:
            print(f"Warning: No DE_time found in {file_path}")
            return False

        # Extract primary data
        prefix = keys[0].split('_')[0] # e.g., X097
        
        data_dict = {}
        # Possible sensors: DE, FE, BA
        for sensor in ['DE', 'FE', 'BA']:
            key = f"{prefix}_{sensor}_time"
            if key in mat:
                data_dict[f"vibration_{sensor.lower()}"] = mat[key].flatten()
            elif f"{prefix[1:]}_{sensor}_time" in mat: # Strip the 'X'
                data_dict[f"vibration_{sensor.lower()}"] = mat[f"{prefix[1:]}_{sensor}_time"].flatten()

        # Extract RPM
        rpm_key = f"{prefix}RPM"
        if rpm_key in mat:
            data_dict["rpm"] = float(mat[rpm_key][0][0])
        elif f"{prefix[1:]}RPM" in mat:
            data_dict["rpm"] = float(mat[f"{prefix[1:]}RPM"][0][0])
        else:
            data_dict["rpm"] = 0.0 # Default if not found

        df = pd.DataFrame(data_dict)
        
        # Add metadata as columns (constant for the whole file)
        fault, diameter, load, sampling_rate = metadata
        df['fault_type'] = fault
        df['fault_diameter_inch'] = diameter
        df['load_hp'] = load
        df['sampling_rate_hz'] = 12000 if '12k' in sampling_rate or 'normal' in sampling_rate else 48000
        
        # Save to Parquet
        df.to_parquet(dest_path, index=False)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("Starting raw to interim preprocessing...")
    os.makedirs(INTERIM_DATA_PATH, exist_ok=True)
    
    counts = {"success": 0, "failed": 0}
    
    for category in os.listdir(RAW_DATA_PATH):
        cat_path = os.path.join(RAW_DATA_PATH, category)
        if not os.path.isdir(cat_path):
            continue
            
        interim_cat_path = os.path.join(INTERIM_DATA_PATH, category)
        os.makedirs(interim_cat_path, exist_ok=True)
        
        for filename in os.listdir(cat_path):
            if not filename.endswith('.mat'):
                continue
                
            file_path = os.path.join(cat_path, filename)
            dest_name = filename.replace('.mat', '.parquet')
            dest_path = os.path.join(interim_cat_path, dest_name)
            
            metadata = get_mat_metadata(filename, category)
            success = preprocess_mat_file(file_path, dest_path, metadata)
            
            if success:
                counts["success"] += 1
            else:
                counts["failed"] += 1

    print(f"Preprocessing completed. Success: {counts['success']}, Failed: {counts['failed']}")

if __name__ == "__main__":
    main()
