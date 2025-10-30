"""
Multi-Target Multimodal Training Progress Monitor
Real-time monitoring of training progress, metrics, and system resources
"""

import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')


class TrainingMonitor:
    """
    Monitor training progress for multi-target multimodal model
    """
    
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        self.start_time = None
        self.last_epoch = 0
        
    def check_training_status(self):
        """Check if training is currently running"""
        python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                           if 'python' in p.info['name'].lower()]
        
        training_running = False
        for proc in python_processes:
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train_multimodal_multitarget.py' in cmdline:
                    training_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return training_running, python_processes
    
    def get_system_resources(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Check for GPU usage (if available)
        gpu_usage = "N/A"
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = f"{gpu_info.gpu}%"
        except:
            gpu_usage = "Not available"
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'gpu_usage': gpu_usage
        }
    
    def check_output_files(self):
        """Check for training output files"""
        files_status = {}
        
        # Check models directory
        if self.models_dir.exists():
            model_files = list(self.models_dir.glob("*.pt"))
            files_status['models'] = {
                'exists': True,
                'count': len(model_files),
                'files': [f.name for f in model_files],
                'latest': max(model_files, key=os.path.getctime).name if model_files else None
            }
        else:
            files_status['models'] = {'exists': False}
        
        # Check results directory
        if self.results_dir.exists():
            result_files = list(self.results_dir.glob("*"))
            files_status['results'] = {
                'exists': True,
                'count': len(result_files),
                'files': [f.name for f in result_files]
            }
        else:
            files_status['results'] = {'exists': False}
        
        return files_status
    
    def estimate_progress(self):
        """Estimate training progress based on files and time"""
        if not self.start_time:
            self.start_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        estimated_total_time = 4 * 3600  # 4 hours in seconds
        progress_percent = min((elapsed_time / estimated_total_time) * 100, 95)
        
        return {
            'elapsed_hours': elapsed_time / 3600,
            'estimated_progress': progress_percent,
            'estimated_remaining_hours': max(0, (estimated_total_time - elapsed_time) / 3600)
        }
    
    def create_progress_report(self):
        """Create a comprehensive progress report"""
        print("=" * 80)
        print("🚀 MULTI-TARGET MULTIMODAL TRAINING PROGRESS REPORT")
        print("=" * 80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check training status
        training_running, python_processes = self.check_training_status()
        print(f"\n🔄 Training Status: {'✅ RUNNING' if training_running else '❌ NOT RUNNING'}")
        
        if training_running:
            print(f"   • Python processes: {len(python_processes)}")
            for proc in python_processes:
                try:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'train_multimodal_multitarget.py' in cmdline:
                        print(f"   • Training PID: {proc.info['pid']}")
                        break
                except:
                    continue
        
        # System resources
        print(f"\n💻 System Resources:")
        resources = self.get_system_resources()
        print(f"   • CPU Usage: {resources['cpu_percent']:.1f}%")
        print(f"   • Memory: {resources['memory_percent']:.1f}% ({resources['memory_used_gb']:.1f}GB / {resources['memory_total_gb']:.1f}GB)")
        print(f"   • GPU Usage: {resources['gpu_usage']}")
        
        # Output files
        print(f"\n📁 Output Files:")
        files_status = self.check_output_files()
        
        if files_status['models']['exists']:
            print(f"   • Models: {files_status['models']['count']} files")
            if files_status['models']['latest']:
                print(f"   • Latest: {files_status['models']['latest']}")
        else:
            print(f"   • Models: No files yet")
        
        if files_status['results']['exists']:
            print(f"   • Results: {files_status['results']['count']} files")
        else:
            print(f"   • Results: No files yet")
        
        # Progress estimation
        print(f"\n⏱️ Progress Estimation:")
        progress = self.estimate_progress()
        print(f"   • Elapsed: {progress['elapsed_hours']:.1f} hours")
        print(f"   • Estimated Progress: {progress['estimated_progress']:.1f}%")
        print(f"   • Estimated Remaining: {progress['estimated_remaining_hours']:.1f} hours")
        
        # Training details
        print(f"\n📊 Training Details:")
        print(f"   • Dataset: 500,000 donors")
        print(f"   • Features: 24 tabular + sequence + network")
        print(f"   • Targets: 5 engagement behaviors")
        print(f"   • Model: 4.2M parameters with GNN")
        print(f"   • Device: CUDA (GPU)")
        
        print("\n" + "=" * 80)
        
        return {
            'training_running': training_running,
            'resources': resources,
            'files': files_status,
            'progress': progress
        }
    
    def monitor_continuously(self, interval=60):
        """Monitor training continuously with specified interval"""
        print(f"🔄 Starting continuous monitoring (every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                self.create_progress_report()
                print(f"\n⏳ Next check in {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def create_progress_visualization(self):
        """Create a progress visualization"""
        if not self.results_dir.exists():
            print("❌ No results directory found for visualization")
            return
        
        # This would create visualizations if we had training history data
        print("📊 Progress visualization would be created here")
        print("   (Requires training history data from completed epochs)")


def main():
    """Main monitoring function"""
    monitor = TrainingMonitor()
    
    print("🔍 Multi-Target Multimodal Training Monitor")
    print("=" * 50)
    
    # Create single report
    report = monitor.create_progress_report()
    
    # Ask if user wants continuous monitoring
    print(f"\n❓ Would you like continuous monitoring? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes']:
            interval = 60  # 1 minute
            print(f"⏱️ Enter monitoring interval in seconds (default {interval}): ", end="")
            try:
                interval = int(input()) or interval
            except:
                pass
            monitor.monitor_continuously(interval)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
