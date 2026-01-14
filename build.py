import os
import sys
import platform
import subprocess
import shutil

def run_command(command):
    try:
        print(f"\n[Command]: {' '.join(command)}\n")
        result = subprocess.run(command, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

def main():
    system_os = platform.system()
    
    common_args = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--enable-plugin=pyside6",
        "--assume-yes-for-downloads",
        "--include-data-dir=fonts=fonts",
        "--include-data-dir=models=models",
        "--include-data-dir=assets=assets",
        "--include-data-file=version.json=version.json",
        "main.py"
    ]

    print(f"Detected OS: {system_os}")

    if system_os == "Windows":
        # === 2. Windows 설정 (Onefile) ===
        print("Starting build for Windows (Onefile)...")
        
        windows_args = [
            "--onefile",
            "--windows-console-mode=disable",
            "--windows-icon-from-ico=assets/icon.ico",
            "--output-filename=ScoreCapturePro.exe"
        ]
        
        final_cmd = common_args[:-1] + windows_args + [common_args[-1]]
        
        run_command(final_cmd)
        print("\nBuild Complete! Check 'ScoreCapturePro.exe'")

    elif system_os == "Darwin":
        # === 3. macOS 설정 (App Bundle) ===
        print("Starting build for macOS (App Bundle)...")
        
        macos_args = [
            "--macos-create-app-bundle",
            "--macos-disable-console",
            "--macos-app-icon=assets/icon.png",
            "--output-filename=ScoreCapturePro"
        ]
        
        final_cmd = common_args[:-1] + macos_args + [common_args[-1]]
        
        run_command(final_cmd)
        
        expected_app = "ScoreCapturePro.app"
        default_output = "main.app"
        
        if os.path.exists(default_output) and not os.path.exists(expected_app):
            print(f"Renaming {default_output} to {expected_app}...")
            shutil.move(default_output, expected_app)
            
        zip_name = "ScoreCapturePro_Mac.zip"
        
        if os.path.exists(expected_app):
            print(f"Zipping {expected_app} to {zip_name}...")
            subprocess.run(["zip", "-r", zip_name, expected_app], check=True)
            print(f"\nBuild Complete! Check '{zip_name}'")
        else:
            print(f"Error: {expected_app} not found. Build might have failed.")
            sys.exit(1)

    else:
        print(f"Unsupported OS: {system_os}")
        sys.exit(1)

if __name__ == "__main__":
    print("Installing build dependencies...")
    build_deps = ["nuitka", "zstandard"]
    if sys.platform == "darwin":
        build_deps.append("pyobjc-framework-Quartz")
        
    subprocess.run([sys.executable, "-m", "pip", "install"] + build_deps, check=False)
    
    main()