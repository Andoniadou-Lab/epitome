import os
import csv
from datetime import datetime
import streamlit as st
from config import Config
import uuid

def get_session_id():
    """
    Get or create a unique session ID for the current user.
    
    Returns:
    --------
    str
        A unique session identifier
    """
    if 'session_id' not in st.session_state:
        # Generate a UUID for the session
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def add_activity(value, analysis, user=None, time=None, file="analytics.txt"):
    """
    Add an activity log entry to the analytics file.
    
    Parameters:
    -----------
    value : str
        The value or data being analyzed
    analysis : str
        Type of analysis being performed
    user : str, optional
        User identifier. If None, uses session ID
    time : str or datetime, optional
        Timestamp for the activity. If None, current time is used
    file : str, optional
        Name of the analytics file (default: "analytics.txt")
    """
    # Use Config.BASE_PATH to ensure proper directory structure
    base_path = Config.BASE_PATH
    
    # Create data directory path
    data_dir = os.path.join(base_path, "data")
    
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Complete file path
    file_path = os.path.join(data_dir, file)
    
    # If time is not provided, use current time
    if time is None:
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert time to string if it's a datetime object
    elif isinstance(time, datetime):
        time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # If user is not provided, use session ID
    if user is None:
        user = get_session_id()
    
    # Check if file exists
    file_exists = os.path.exists(file_path)
    
    # Open file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['value', 'analysis', 'user', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the activity data
        writer.writerow({
            'value': value,
            'analysis': analysis,
            'user': user,
            'time': time
        })