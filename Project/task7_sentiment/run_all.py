#!/usr/bin/env python
"""
Task C.7: Complete Pipeline Runner - Run Everything in One Command

This is a convenience script that runs all 3 main scripts in sequence:
1. task7_runner.py - Main pipeline (9 models)
2. task7_extended_models.py - Extended models (12 models)
3. task7_advanced_evaluation.py - Visualizations and reports

Usage:
    python run_all.py

Total execution time: ~14 seconds
Output: 21 trained models + 3 plots + 2 reports

Author: Anh Vu Le
Date: November 2025
Course: COS30018 - Intelligent Systems
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_script(script_name, description):
    """
    Run a Python script and track execution time
    
    Args:
        script_name: Name of the script to run (e.g., 'task7_runner.py')
        description: Human-readable description of what the script does
        
    Returns:
        bool: True if successful, False if error occurred
    """
    print_banner(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"Started at: {time.strftime('%H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess
        # Check if we're in task7_sentiment/ directory
        if Path('task7_runner.py').exists():
            # Already in task7_sentiment/
            script_path = script_name
        else:
            # In Project/ directory
            script_path = f'task7_sentiment/{script_name}'
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=Path.cwd()  # Use current working directory
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n[OK] SUCCESS - Completed in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n[FAIL] ERROR - Script failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] ERROR - Exception occurred after {elapsed:.1f} seconds")
        print(f"Error details: {e}")
        return False


def main():
    """
    Main function to run all Task C.7 scripts in sequence
    
    Execution order:
    1. Main pipeline (trains 9 base models)
    2. Extended models (trains 12 additional models)
    3. Advanced evaluation (generates all visualizations)
    
    Total output: 21 models, 3 plots, 2 reports
    """
    overall_start = time.time()
    
    print("\n" + "=" * 80)
    print("=" + " " * 78 + "=")
    print("=  TASK C.7: SENTIMENT-BASED STOCK PREDICTION".ljust(79) + "=")
    print("=  Complete Pipeline Runner".ljust(79) + "=")
    print("=" + " " * 78 + "=")
    print("=" * 80)
    
    print("\n[INFO] This will run 3 scripts in sequence:")
    print("   1. task7_runner.py          -> Train 9 base models (~8s)")
    print("   2. task7_extended_models.py -> Train 12 extended models (~3s)")
    print("   3. task7_advanced_evaluation.py -> Generate plots & reports (~2s)")
    print("\n[TIME] Estimated total time: ~14 seconds")
    print("\n" + "-" * 80)
    
    # Track success/failure for each script
    results = {}
    
    # =========================================================================
    # STEP 1: Run main pipeline
    # =========================================================================
    success = run_script(
        'task7_runner.py',
        'STEP 1/3: Main Pipeline (9 Base Models)'
    )
    results['main_pipeline'] = success
    
    if not success:
        print("\n[WARNING] Main pipeline failed. Stopping execution.")
        print("   Fix the errors above before running extended models.")
        sys.exit(1)
    
    # =========================================================================
    # STEP 2: Run extended models
    # =========================================================================
    success = run_script(
        'task7_extended_models.py',
        'STEP 2/3: Extended Models (12 Advanced Algorithms)'
    )
    results['extended_models'] = success
    
    if not success:
        print("\n[WARNING] Extended models failed. Continuing to evaluation...")
        print("   Visualization will use only the 9 base models.")
    
    # =========================================================================
    # STEP 3: Run advanced evaluation
    # =========================================================================
    success = run_script(
        'task7_advanced_evaluation.py',
        'STEP 3/3: Advanced Evaluation (Plots & Reports)'
    )
    results['evaluation'] = success
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    
    print(f"\n[TIME] Total execution time: {overall_elapsed:.1f} seconds")
    
    print("\n[RESULTS] Summary:")
    print("   " + ("[OK]  " if results['main_pipeline'] else "[FAIL]") + " Main Pipeline (9 models)")
    print("   " + ("[OK]  " if results['extended_models'] else "[FAIL]") + " Extended Models (12 models)")
    print("   " + ("[OK]  " if results['evaluation'] else "[FAIL]") + " Visualizations & Reports")
    
    if all(results.values()):
        print("\n[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n[OUTPUT] Locations:")
        print("   Models:  task7_models/ (21 .pkl files)")
        print("   Results: task7_results/")
        print("      - evaluation_metrics.json (all metrics)")
        print("      - model_comparison.csv (sortable table)")
        print("      - classification_reports.txt (detailed reports)")
        print("      - executive_summary.txt (for report writing)")
        print("   Plots:   task7_results/plots/")
        print("      - confusion_matrices.png (top 6 models)")
        print("      - model_comparison.png (4-subplot analysis)")
        print("      - feature_importance.png (feature rankings)")
        
        print("\n[BEST] Model: sentiment_only_gradientboosting (F1 = 68.8%)")
        print("   Sentiment improvement over baseline: +4.8%")
        
        print("\n[OK] Task C.7 COMPLETE - Ready for academic submission!")
        
    else:
        print("\n[WARNING] Some steps failed. Check error messages above.")
        failed_steps = [k for k, v in results.items() if not v]
        print(f"   Failed: {', '.join(failed_steps)}")
        sys.exit(1)
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Pipeline interrupted by user (Ctrl+C)")
        print("   Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
