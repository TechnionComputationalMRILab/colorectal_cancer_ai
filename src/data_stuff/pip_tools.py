import subprocess
import sys

def install(packages: list, quietly: bool=True):
    call_list = [sys.executable, "-m", "pip", "install", *packages]
    if quietly:
        call_list = [sys.executable, "-m", "pip", "install", "-q", *packages]
        print("🤫 Installing packages quietly... shhh 🤫")
    subprocess.check_call(call_list)
    print("✅ Done installing packages")
