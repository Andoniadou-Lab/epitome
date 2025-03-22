import subprocess
import os
import datetime
from config import Config
def save_package_versions():
    """
    Capture currently installed Python packages using pip freeze
    and save them to versions.txt in the base path with a timestamp.
    """
    BASE_PATH = Config.BASE_PATH
    
    # Create the output file path
    output_file = os.path.join(BASE_PATH, 'code', 'versions.txt')
    
    try:
        # Run pip freeze and capture the output
        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Process the output to simplify it
        raw_packages = result.stdout.strip().split('\n')
        simplified_packages = []
        
        for pkg in raw_packages:
            # Remove any environment markers or comments
            if '==' in pkg:
                # Take just the package name and version
                pkg_parts = pkg.split('==')
                if len(pkg_parts) >= 2:
                    pkg_name = pkg_parts[0]
                    # Remove any trailing semicolon and environment markers
                    pkg_version = pkg_parts[1].split(';')[0].strip()
                    simplified_packages.append(f"{pkg_name}=={pkg_version}")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write the simplified output to the versions.txt file
        with open(output_file, 'w') as f:
            f.write(f"# Epitome Package Versions\n")
            f.write(f"# Captured on: {timestamp}\n\n")
            f.write('\n'.join(simplified_packages))
        
        print(f"✓ Successfully saved package versions to {output_file}")
        print(f"✓ Captured {len(simplified_packages)} packages")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running pip freeze: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Error saving package versions: {str(e)}")
        return False

if __name__ == "__main__":
    print("Capturing current Python package environment...")
    success = save_package_versions()
    print("Done" if success else "Failed")