#!/usr/bin/env python3
"""
Test runner script for Connect4 RL project.

Provides different test execution configurations and reporting options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    result = subprocess.run(cmd, capture_output=False, cwd=str(project_root))
    
    if result.returncode != 0:
        print(f"\n[FAIL] {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] {description} completed successfully")
        return True


def run_all_tests():
    """Run all tests without generating coverage files."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "All tests")


def run_unit_tests():
    """Run only unit tests (fast tests)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest", 
        str(project_root / "tests"),
        "-m", "not slow and not integration and not performance",
        "-v"
    ]
    return run_command(cmd, "Unit tests only")


def run_integration_tests():
    """Run integration tests."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-m", "integration",
        "-v",
        "--timeout=300"
    ]
    return run_command(cmd, "Integration tests")


def run_performance_tests():
    """Run performance and benchmark tests."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-m", "performance",
        "-v",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ]
    return run_command(cmd, "Performance tests")


def run_gpu_tests():
    """Run GPU-specific tests."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-m", "gpu",
        "-v"
    ]
    return run_command(cmd, "GPU tests")


def run_parallel_tests():
    """Run tests in parallel."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-n", "auto",
        "-v"
    ]
    return run_command(cmd, "Parallel test execution")


def run_specific_test_file(test_file):
    """Run tests from a specific file."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests" / test_file),
        "-v"
    ]
    return run_command(cmd, f"Tests from {test_file}")


def run_specific_test_category(category):
    """Run tests from a specific category."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    valid_categories = [
        "test_environments", "test_agents", "test_training", 
        "test_core", "test_utils", "test_scripts", 
        "test_performance", "test_integration"
    ]
    
    if category not in valid_categories:
        print(f"[ERROR] Invalid category: {category}")
        print(f"Valid categories: {', '.join(valid_categories)}")
        return False
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests" / category),
        "-v"
    ]
    return run_command(cmd, f"Tests from {category}")


def check_test_setup():
    """Check that test environment is properly set up."""
    print("[CHECK] Checking test environment setup...")
    
    # Get project root directory (parent of scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Change to project root directory
    original_cwd = os.getcwd()
    os.chdir(str(project_root))
    
    try:
        # Check if we're in the right directory structure
        if not (project_root / "tests").exists():
            print(f"[ERROR] Tests directory not found. Expected at {project_root / 'tests'}")
            return False
        
        # Check if pytest is installed
        try:
            import pytest
            print(f"[OK] pytest version: {pytest.__version__}")
        except ImportError:
            print("[ERROR] pytest not installed. Run: pip install -r requirements-test.txt")
            return False
        
        # Check if source code is importable
        try:
            # Add project root to Python path so src module can be found
            project_root_str = str(project_root)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
            
            # Try to import a simple source module
            import src.environments.connect4_game
            print("[OK] Source code is importable")
        except ImportError as e:
            print(f"[WARN] Import issue detected: {e}")
            print("[INFO] This may not affect test execution if tests handle imports properly")
            # Don't return False here as tests might still work
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    # Check if CUDA is available (optional)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"[INFO] CUDA available: {cuda_available}")
        if cuda_available:
            print(f"[INFO] CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("[WARN] PyTorch not available - some tests may be skipped")
    
    print("[OK] Test environment setup looks good!")
    return True


def run_coverage_tests():
    """Run all tests with coverage reporting (creates files)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("[WARNING] This will create coverage files (htmlcov/, coverage.xml)")
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        "-v",
        f"--cov={project_root / 'src'}",
        f"--cov-report=html:{project_root / 'htmlcov'}",
        "--cov-report=term-missing",
        f"--cov-report=xml:{project_root / 'coverage.xml'}",
        "--cov-fail-under=85"
    ]
    return run_command(cmd, "All tests with coverage")


def generate_test_report():
    """Generate comprehensive test report (creates multiple files)."""
    print("[REPORT] Generating comprehensive test report...")
    print("[WARNING] This will create: htmlcov/, coverage.xml, report.html, test_report.json")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        "python", "-m", "pytest",
        str(project_root / "tests"),
        f"--cov={project_root / 'src'}",
        f"--cov-report=html:{project_root / 'htmlcov'}",
        f"--cov-report=xml:{project_root / 'coverage.xml'}",
        f"--html={project_root / 'report.html'}",
        "--self-contained-html",
        "--json-report",
        f"--json-report-file={project_root / 'test_report.json'}",
        "-v"
    ]
    
    success = run_command(cmd, "Comprehensive test report generation")
    
    if success:
        print("\n[RESULTS] Report files generated:")
        print("  - htmlcov/index.html (Coverage report)")
        print("  - report.html (Test results)")
        print("  - test_report.json (JSON report)")
        print("  - coverage.xml (XML coverage)")
    
    return success


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for Connect4 RL project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                 # Run all tests (no files created)
  python run_tests.py --coverage            # Run tests with coverage (creates files)
  python run_tests.py --unit                # Run only unit tests
  python run_tests.py --integration         # Run integration tests
  python run_tests.py --performance         # Run performance tests
  python run_tests.py --gpu                 # Run GPU tests
  python run_tests.py --parallel            # Run tests in parallel
  python run_tests.py --category agents     # Run agent tests
  python run_tests.py --file test_ppo_agent.py  # Run specific file
  python run_tests.py --check               # Check test setup
  python run_tests.py --report              # Generate comprehensive report (creates files)
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests (no files created)")
    parser.add_argument("--coverage", action="store_true",
                       help="Run all tests with coverage (creates htmlcov/, coverage.xml)")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only (fast)")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    parser.add_argument("--gpu", action="store_true",
                       help="Run GPU tests")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("--category", type=str,
                       help="Run tests from specific category")
    parser.add_argument("--file", type=str,
                       help="Run specific test file")
    parser.add_argument("--check", action="store_true",
                       help="Check test environment setup")
    parser.add_argument("--report", action="store_true",
                       help="Generate comprehensive test report (creates multiple files)")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 1
    
    success = True
    
    if args.check:
        success &= check_test_setup()
    
    if args.all:
        success &= run_all_tests()
    
    if args.coverage:
        success &= run_coverage_tests()
    
    if args.unit:
        success &= run_unit_tests()
    
    if args.integration:
        success &= run_integration_tests()
    
    if args.performance:
        success &= run_performance_tests()
    
    if args.gpu:
        success &= run_gpu_tests()
    
    if args.parallel:
        success &= run_parallel_tests()
    
    if args.category:
        success &= run_specific_test_category(args.category)
    
    if args.file:
        success &= run_specific_test_file(args.file)
    
    if args.report:
        success &= generate_test_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())