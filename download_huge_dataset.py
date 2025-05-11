import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import tarfile

def create_target_dir():
    target_dir = Path("huge_dataset")
    target_dir.mkdir(exist_ok=True)
    return target_dir

def download_package(package_name, target_dir):
    try:
        # Use pip download to get the package without installing
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--no-deps",  # Don't download dependencies
            "--no-binary", ":all:",  # Download source distributions
            "--dest", str(target_dir),
            package_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded {package_name}")
        else:
            print(f"Failed to download {package_name}: {result.stderr}")
            
    except Exception as e:
        print(f"Error processing {package_name}: {str(e)}")

def extract_targz_files(target_dir):
    print("Extracting .tar.gz files...")
    target_path = Path(target_dir)
    
    # Find all .tar.gz files
    targz_files = list(target_path.glob("*.tar.gz"))
    
    for targz_file in tqdm(targz_files, desc="Extracting"):
        try:
            # Create a directory with the same name as the tar file (without .tar.gz)
            extract_dir = target_path / targz_file.stem.replace('.tar', '')
            extract_dir.mkdir(exist_ok=True)
            
            # Extract the tar file
            with tarfile.open(targz_file, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            # Remove the original .tar.gz file
            targz_file.unlink()
            
        except Exception as e:
            print(f"Error extracting {targz_file.name}: {str(e)}")

def main():
    # Create target directory
    target_dir = create_target_dir()
    
    # Read package list
    with open("huge_dataset_list.txt", "r") as f:
        packages = [line.strip() for line in f if line.strip()]
    
    # Process each package
    # for package in tqdm(packages):
    #     download_package(package, target_dir)
    
    # Extract all downloaded .tar.gz files
    extract_targz_files(target_dir)

if __name__ == "__main__":
    main() 