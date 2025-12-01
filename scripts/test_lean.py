#!/usr/bin/env python3
"""
Test Lean code using the LeanREPL.

This script takes Lean code (from a file, stdin, or as an argument),
sends it to the LeanREPL, and reports whether it compiles successfully
and makes sense.

Usage:
    python test_lean.py --code "def f := 2"
    python test_lean.py --file path/to/file.lean
    echo "def f := 2" | python test_lean.py
    
    # Or import as a module:
    from scripts.test_lean import test_lean_code
    result = test_lean_code("def f := 2")
    print(result["success"])
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_repl_executable(project_root: Path) -> Optional[Path]:
    """
    Find the LeanREPL executable.
    
    Checks multiple possible locations:
    1. LeanREPL/build/bin/LeanREPL (relative to project root)
    2. Try building it if it doesn't exist
    """
    leanrepl_dir = project_root / "LeanREPL"
    leanrepl_executable = leanrepl_dir / "build" / "bin" / "LeanREPL"
    
    if leanrepl_executable.exists() and os.access(leanrepl_executable, os.X_OK):
        return leanrepl_executable
    
    # Try building it if it doesn't exist
    if leanrepl_dir.exists():
        # Ensure elan is in PATH
        env = os.environ.copy()
        elan_path = Path.home() / ".elan" / "bin"
        if elan_path.exists():
            env["PATH"] = str(elan_path) + os.pathsep + env.get("PATH", "")
        
        try:
            result = subprocess.run(
                ["lake", "build"],
                cwd=leanrepl_dir,
                capture_output=True,
                timeout=60,
                env=env
            )
            if result.returncode == 0 and leanrepl_executable.exists():
                return leanrepl_executable
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return None


def run_repl_command(lean_code: str, repl_executable: Path, repl_dir: Path, timeout: int = 30) -> Tuple[str, int]:
    """
    Send Lean code to the LeanREPL and get the response.
    
    Args:
        lean_code: The Lean code to test
        repl_executable: Path to the LeanREPL executable
        repl_dir: Directory containing the LeanREPL (unused, kept for compatibility)
        timeout: Timeout in seconds for the REPL command
        
    Returns:
        Tuple of (response_string, return_code)
    """
    # LeanREPL expects plain text commands, not JSON
    # We send the command followed by !quit to exit
    input_text = lean_code + "\n!quit\n"
    
    try:
        # Use direct executable path
        # Set environment to ensure Lean toolchain is accessible
        env = os.environ.copy()
        # Ensure elan is in PATH if it exists in home directory
        elan_path = Path.home() / ".elan" / "bin"
        if elan_path.exists():
            env["PATH"] = str(elan_path) + os.pathsep + env.get("PATH", "")
        
        # Set LEAN_PATH to the correct toolchain
        # Read the lean-toolchain file to determine which toolchain to use
        leanrepl_dir = repl_executable.parent.parent.parent  # Go from build/bin/LeanREPL to LeanREPL/
        toolchain_file = leanrepl_dir / "lean-toolchain"
        if toolchain_file.exists():
            toolchain_spec = toolchain_file.read_text().strip()
            # Find the toolchain directory
            # Format is like "leanprover/lean4:v4.26.0-rc2" or "leanprover/lean4-nightly:nightly-2022-07-31"
            toolchain_name = toolchain_spec.replace("/", "--").replace(":", "---")
            toolchain_dir = Path.home() / ".elan" / "toolchains" / toolchain_name
            if toolchain_dir.exists():
                lean_lib_path = toolchain_dir / "lib" / "lean"
                if lean_lib_path.exists():
                    env["LEAN_PATH"] = str(lean_lib_path)
        
        process = subprocess.Popen(
            [str(repl_executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=env
        )
        
        stdout, _ = process.communicate(input=input_text, timeout=timeout)
        return_code = process.returncode
        
        return stdout.strip(), return_code
        
    except subprocess.TimeoutExpired:
        process.kill()
        return f"REPL timeout (exceeded {timeout} seconds)", -1
    except Exception as e:
        return f"Error running REPL: {str(e)}", -1


def parse_repl_response(response: str) -> Dict:
    """
    Parse the LeanREPL text response and extract important information.
    
    LeanREPL outputs plain text with error messages in the format:
    "repl:1:X: error: <message>"
    
    Returns a dictionary with:
    - success: bool (True if no errors)
    - has_errors: bool
    - has_warnings: bool
    - error_count: int
    - warning_count: int
    - errors: list of error messages
    - warnings: list of warning messages
    - has_sorries: bool (whether the code contains sorries)
    - env: optional environment identifier (not used by LeanREPL)
    - raw_response: the full response text
    """
    result = {
        "success": True,
        "has_errors": False,
        "has_warnings": False,
        "error_count": 0,
        "warning_count": 0,
        "errors": [],
        "warnings": [],
        "has_sorries": False,
        "env": None,
        "raw_response": response
    }
    
    if not response:
        result["success"] = False
        result["errors"].append("Empty response from REPL")
        return result
    
    # Parse text output for errors and warnings
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip the prompt "> " (LeanREPL prints this before each command)
        if line == ">" or line.startswith("> "):
            continue
        
        # Check for error patterns: "repl:1:X: error: ..." or similar
        # LeanREPL outputs errors in format like "repl:1:4: error: <message>"
        if ": error:" in line.lower():
            result["has_errors"] = True
            result["error_count"] += 1
            # Extract error message (everything after "error:")
            error_msg = line.split("error:", 1)[-1].strip()
            if error_msg:
                result["errors"].append(error_msg)
            else:
                result["errors"].append(line)
        
        # Check for warning patterns: "repl:1:X: warning: ..."
        elif ": warning:" in line.lower():
            result["has_warnings"] = True
            result["warning_count"] += 1
            # Extract warning message (everything after "warning:")
            warning_msg = line.split("warning:", 1)[-1].strip()
            if warning_msg:
                result["warnings"].append(warning_msg)
            else:
                result["warnings"].append(line)
        
        # Check for "sorry" in the response (incomplete proofs)
        if "sorry" in line.lower():
            result["has_sorries"] = True
    
    # Check for toolchain/installation errors
    if "invalid header" in response.lower() or "failed to read file" in response.lower():
        result["has_errors"] = True
        result["error_count"] += 1
        if "invalid header" in response.lower():
            result["errors"].append(
                "Lean toolchain file appears corrupted or version mismatch. "
                "Try rebuilding LeanREPL: cd LeanREPL && lake build"
            )
        else:
            result["errors"].append("Failed to read Lean toolchain file. Check Lean installation.")
    
    # Also check for common error indicators in the raw response
    if "error" in response.lower() and not result["has_errors"]:
        # If we see "error" but didn't catch it with the pattern, add it
        result["has_errors"] = True
        result["error_count"] += 1
        result["errors"].append("Error detected in REPL output (see raw_response)")
    
    # Determine overall success
    result["success"] = not result["has_errors"]
    
    return result


def test_lean_code(lean_code: str, project_root: Optional[Path] = None, timeout: int = 30, verbose: bool = False) -> Dict:
    """
    Test Lean code and return results.
    
    Args:
        lean_code: The Lean code to test
        project_root: Root directory of the project (will auto-detect if None)
        timeout: Timeout in seconds for REPL command
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with test results (from parse_repl_response)
    """
    # Find project root if not provided
    if project_root is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
    
    # Find LeanREPL executable
    leanrepl_dir = project_root / "LeanREPL"
    repl_executable = find_repl_executable(project_root)
    
    if repl_executable is None:
        return {
            "success": False,
            "has_errors": True,
            "errors": [f"Could not find LeanREPL executable. Tried {leanrepl_dir / 'build' / 'bin' / 'LeanREPL'}. "
                      "Please ensure LeanREPL is built by running 'lake build' in the LeanREPL directory."],
            "raw_response": None
        }
    
    if verbose:
        print(f"Using LeanREPL: {repl_executable}", file=sys.stderr)
        print(f"LeanREPL directory: {leanrepl_dir}", file=sys.stderr)
    
    # Run the REPL command
    response, return_code = run_repl_command(lean_code, repl_executable, leanrepl_dir, timeout=timeout)
    
    if verbose:
        print(f"REPL return code: {return_code}", file=sys.stderr)
        print(f"REPL response: {response[:500]}...", file=sys.stderr)
    
    # Parse the response
    result = parse_repl_response(response)
    
    # Add return code information
    if return_code != 0:
        result["success"] = False
        result["has_errors"] = True
        result["errors"].append(f"REPL process exited with code {return_code}")
        result["raw_response"] = response
    
    return result


def format_output(result: Dict, format: str = "human") -> str:
    """
    Format the test results for output.
    
    Args:
        result: Dictionary from parse_repl_response
        format: Output format ("human", "json", "brief")
        
    Returns:
        Formatted string
    """
    if format == "json":
        return json.dumps(result, indent=2)
    
    if format == "brief":
        if result["success"]:
            status = "✓ PASS"
            if result["has_sorries"]:
                status += " (with sorries)"
            return status
        else:
            return "✗ FAIL"
    
    # Human-readable format
    lines = []
    
    if result["success"]:
        lines.append("✓ COMPILES SUCCESSFULLY")
    else:
        lines.append("✗ COMPILATION FAILED")
    
    if result["has_errors"]:
        lines.append(f"\nErrors ({result['error_count']}):")
        for i, error in enumerate(result["errors"], 1):
            lines.append(f"  {i}. {error}")
    
    if result["has_warnings"]:
        lines.append(f"\nWarnings ({result['warning_count']}):")
        for i, warning in enumerate(result["warnings"], 1):
            lines.append(f"  {i}. {warning}")
    
    if result["has_sorries"]:
        lines.append("\n⚠ Contains 'sorry' (incomplete proof)")
    
    if result["env"] is not None:
        lines.append(f"\nEnvironment ID: {result['env']}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Test Lean code using the LeanREPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test code from command line
  python test_lean.py --code "def f := 2"

  # Test code from file
  python test_lean.py --file example.lean

  # Test code from stdin
  echo "def f := 2" | python test_lean.py

  # JSON output format
  python test_lean.py --code "def f := 2" --format json

  # Brief output format
  python test_lean.py --code "def f := 2" --format brief
        """
    )
    
    parser.add_argument(
        "--code",
        type=str,
        help="Lean code to test (as a string)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file containing Lean code"
    )
    
    parser.add_argument(
        "--format",
        choices=["human", "json", "brief"],
        default="human",
        help="Output format (default: human)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for REPL command (default: 30)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debugging information"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        help="Root directory of the project (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Get Lean code from one of the sources
    lean_code = None
    
    if args.code:
        lean_code = args.code
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        lean_code = file_path.read_text()
    elif not sys.stdin.isatty():
        # Read from stdin if not a TTY
        lean_code = sys.stdin.read()
    else:
        parser.print_help()
        print("\nError: No Lean code provided. Use --code, --file, or pipe to stdin.", file=sys.stderr)
        sys.exit(1)
    
    if not lean_code or not lean_code.strip():
        print("Error: Empty Lean code provided.", file=sys.stderr)
        sys.exit(1)
    
    # Determine project root
    project_root = None
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
    
    # Test the Lean code
    result = test_lean_code(
        lean_code,
        project_root=project_root,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Output results
    output = format_output(result, format=args.format)
    print(output)
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

