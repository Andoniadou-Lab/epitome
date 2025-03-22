import psutil
import threading
import time
from datetime import datetime
import sys
import atexit

class MemoryTracker:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MemoryTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self, interval=10):
        """
        Initialize memory tracker (Singleton pattern)
        
        Parameters:
        -----------
        interval : int
            Time interval in seconds between memory checks
        """
        if not self._initialized:
            self.interval = interval
            self._stop_event = threading.Event()
            self._thread = None
            self._initialized = True
            atexit.register(self.stop)
            print("Memory tracker initialized", flush=True)

    def _track_memory(self):
        """Monitor memory usage and print to terminal"""
        try:
            process = psutil.Process()
            while not self._stop_event.is_set():
                try:
                    # Get memory usage in MB
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[MEMORY TRACKER][{current_time}] Memory Usage: {memory_usage:.2f} MB", 
                          file=sys.stderr, 
                          flush=True)
                except Exception as e:
                    print(f"\n[MEMORY TRACKER] Error getting memory usage: {str(e)}", 
                          file=sys.stderr, 
                          flush=True)
                time.sleep(self.interval)
        except Exception as e:
            print(f"\n[MEMORY TRACKER] Critical error: {str(e)}", 
                  file=sys.stderr, 
                  flush=True)

    def start(self):
        """Start memory tracking in a separate thread"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._track_memory)
            self._thread.daemon = True
            self._thread.start()
            print("[MEMORY TRACKER] Started tracking", flush=True)

    def stop(self):
        """Stop memory tracking"""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            print("[MEMORY TRACKER] Stopped tracking", flush=True)
            MemoryTracker._instance = None
            self._initialized = False

def initialize_memory_tracker(interval=10):
    """Initialize and start the memory tracker"""
    tracker = MemoryTracker(interval)
    tracker.start()
    return tracker