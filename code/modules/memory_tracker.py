import psutil
import threading
import time
from datetime import datetime
import sys
import atexit
import traceback
import functools
import gc
import os
from contextlib import contextmanager

class MemoryTracker:
    _instance = None
    _initialized = False
    
    # Track memory usage by tab/function
    _tab_memory = {}
    _function_timing = {}
    _current_tab = None
    _start_memory = 0
    _peak_memory = 0
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MemoryTracker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, interval=5):
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
            self._log_file = os.path.join(os.getcwd(), "memory_log.txt")
            
            # Create empty log file or append to existing
            with open(self._log_file, 'a') as f:
                f.write(f"\n\n--- NEW SESSION: {datetime.now()} ---\n\n")
            
            atexit.register(self.stop)
            self._start_memory = self._get_memory_usage()
            self._peak_memory = self._start_memory
            
            print(f"Memory tracker initialized. Logging to: {self._log_file}", flush=True)
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def log_message(self, message, to_console=True):
        """Log a message to file and optionally to console"""
        with open(self._log_file, 'a') as f:
            f.write(f"{message}\n")
        if to_console:
            print(message, file=sys.stderr, flush=True)
    
    def _track_memory(self):
        """Monitor memory usage and log to file and terminal"""
        try:
            process = psutil.Process()
            
            # Log initial system information
            self.log_message(f"\n[SYSTEM INFO] CPU Cores: {psutil.cpu_count()}")
            self.log_message(f"[SYSTEM INFO] Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
            
            while not self._stop_event.is_set():
                try:
                    # Get memory usage in MB
                    memory_usage = process.memory_info().rss / (1024 * 1024)
                    self._peak_memory = max(self._peak_memory, memory_usage)
                    
                    # Get CPU usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    # Get thread count
                    thread_count = len(process.threads())
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    # Check for significant memory increases
                    memory_change = memory_usage - self._start_memory
                    
                    # Format message
                    message = (
                        f"\n[MEMORY TRACKER][{current_time}] "
                        f"Memory: {memory_usage:.2f} MB | "
                        f"Change: {memory_change:+.2f} MB | "
                        f"Peak: {self._peak_memory:.2f} MB | "
                        f"CPU: {cpu_percent:.1f}% | "
                        f"Threads: {thread_count}"
                    )
                    
                    # Add current tab info if available
                    if self._current_tab:
                        message += f" | Tab: {self._current_tab}"
                    
                    self.log_message(message)
                    
                    # Perform garbage collection and log
                    if memory_usage > self._peak_memory * 0.95:  # If close to peak
                        collected = gc.collect()
                        after_gc = self._get_memory_usage()
                        self.log_message(f"[GC] Collected {collected} objects. Memory after: {after_gc:.2f} MB")
                    
                    # Log detailed memory usage by tab
                    if self._tab_memory:
                        tab_info = "\n[TAB MEMORY]:\n" + "\n".join(
                            f"  {tab}: {mem:.2f} MB" for tab, mem in self._tab_memory.items()
                        )
                        self.log_message(tab_info, to_console=False)
                    
                    # Log detailed timing information
                    if self._function_timing:
                        # Sort by total time spent
                        sorted_funcs = sorted(
                            self._function_timing.items(), 
                            key=lambda x: x[1]['total_time'],
                            reverse=True
                        )
                        timing_info = "\n[FUNCTION TIMING] Top 10 functions:\n" + "\n".join(
                            f"  {func}: {stats['total_time']:.2f}s total, {stats['calls']} calls, {stats['avg_time']:.4f}s avg"
                            for func, stats in sorted_funcs[:10]
                        )
                        self.log_message(timing_info, to_console=False)
                    
                except Exception as e:
                    self.log_message(f"\n[MEMORY TRACKER] Error getting memory usage: {str(e)}")
                    self.log_message(traceback.format_exc())
                
                time.sleep(self.interval)
        
        except Exception as e:
            self.log_message(f"\n[MEMORY TRACKER] Critical error: {str(e)}")
            self.log_message(traceback.format_exc())
    
    def register_tab_usage(self, tab_name):
        """Register memory usage for a specific tab"""
        current_memory = self._get_memory_usage()
        self._current_tab = tab_name
        self._tab_memory[tab_name] = current_memory
        self.log_message(f"[TAB SWITCH] Now in tab: {tab_name} | Memory: {current_memory:.2f} MB")
    
    def register_function_call(self, func_name, execution_time):
        """Register timing information for a function"""
        if func_name not in self._function_timing:
            self._function_timing[func_name] = {
                'calls': 0,
                'total_time': 0,
                'avg_time': 0
            }
        
        stats = self._function_timing[func_name]
        stats['calls'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['calls']
        
        # Log long-running functions
        if execution_time > 1.0:  # Log functions taking more than 1 second
            self.log_message(
                f"[SLOW FUNCTION] {func_name} took {execution_time:.2f}s "
                f"(Call #{stats['calls']})"
            )
    
    def start(self):
        """Start memory tracking in a separate thread"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._track_memory)
            self._thread.daemon = True
            self._thread.start()
            self.log_message("[MEMORY TRACKER] Started tracking")
    
    def stop(self):
        """Stop memory tracking"""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            
            # Log summary information
            self.log_message("\n[MEMORY TRACKER] Stopped tracking")
            self.log_message(f"[SUMMARY] Peak memory usage: {self._peak_memory:.2f} MB")
            
            # Memory usage by tab
            if self._tab_memory:
                tab_info = "[SUMMARY] Memory by tab:\n" + "\n".join(
                    f"  {tab}: {mem:.2f} MB" for tab, mem in self._tab_memory.items()
                )
                self.log_message(tab_info)
            
            # Function timing information
            if self._function_timing:
                sorted_funcs = sorted(
                    self._function_timing.items(),
                    key=lambda x: x[1]['total_time'],
                    reverse=True
                )
                timing_info = "[SUMMARY] Top 10 time-consuming functions:\n" + "\n".join(
                    f"  {func}: {stats['total_time']:.2f}s total, {stats['calls']} calls, {stats['avg_time']:.4f}s avg"
                    for func, stats in sorted_funcs[:10]
                )
                self.log_message(timing_info)
            
            MemoryTracker._instance = None
            self._initialized = False


def track_function(func):
    """Decorator to track function execution time and memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            execution_time = end_time - start_time
            memory_change = end_mem - start_mem
            
            # Register with memory tracker if it exists
            tracker = MemoryTracker()
            tracker.register_function_call(func.__name__, execution_time)
            
            # Log significant memory changes
            if abs(memory_change) > 50:  # More than 50MB change
                tracker.log_message(
                    f"[MEMORY CHANGE] {func.__name__} changed memory by {memory_change:+.2f} MB"
                )
    
    return wrapper


@contextmanager
def track_tab(tab_name):
    """Context manager to track which tab is active and measure its impact"""
    tracker = MemoryTracker()
    start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    tracker.register_tab_usage(tab_name)
    
    try:
        yield
    finally:
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        execution_time = end_time - start_time
        memory_change = end_mem - start_mem
        
        tracker.log_message(
            f"[TAB METRICS] '{tab_name}' tab: {execution_time:.2f}s, "
            f"Memory change: {memory_change:+.2f} MB"
        )


def initialize_memory_tracker(interval=5):
    """Initialize and start the memory tracker"""
    tracker = MemoryTracker(interval)
    tracker.start()
    return tracker


# Track important loader functions
def track_specific_functions():
    """Decorate important loading functions with the tracker"""
    import data_loader
    import download
    
    # Apply tracking decorator to these functions
    data_loader.load_and_transform_data = track_function(data_loader.load_and_transform_data)
    data_loader.load_curation_data = track_function(data_loader.load_curation_data)
    data_loader.load_annotation_data = track_function(data_loader.load_annotation_data)
    data_loader.load_motif_data = track_function(data_loader.load_motif_data)
    download.create_downloads_ui_with_metadata = track_function(download.create_downloads_ui_with_metadata)
    download.list_available_h5ad_files = track_function(download.list_available_h5ad_files)