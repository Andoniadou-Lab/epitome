import os
from pathlib import Path

def get_base_path():
    """Get the absolute path to the project root directory."""
    # Check for environment variable first
    return Path(__file__).parent.parent

# Create a Config class to hold all configuration
class Config:
    BASE_PATH = get_base_path()
    AVAILABLE_VERSIONS = ["v_0.01"]  # List of available versions

    @classmethod
    def get_data_path(cls, version, *paths):
        """
        Get the full path to a data file/directory.
        
        Parameters:
        -----------
        version : str
            Version string (e.g., 'v_0.01')
        *paths : str
            Additional path components
            
        Returns:
        --------
        Path
            Full path to the requested resource
        """
        return cls.BASE_PATH / version / os.path.join(*paths)