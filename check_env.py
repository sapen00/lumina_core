import os
from dotenv import load_dotenv

# Print current folder
print(f"Current folder: {os.getcwd()}")

# List all files in this folder
print("\nFiles in this folder:")
files = os.listdir()
for f in files:
    print(f" - {f}")

# Try to load .env
print("\nLoading .env...")
load_dotenv()

# Check variable
uri = os.getenv("MONGODB_URI")
if uri:
    print(f"\nSUCCESS! Found URI: {uri[:15]}...")
else:
    print("\nFAILURE: MONGODB_URI is None.")
    if ".env.txt" in files:
        print(">>> PROBLEM FOUND: You have a file named '.env.txt'. You need to rename it to '.env'!")
    elif ".env" not in files:
        print(">>> PROBLEM FOUND: No '.env' file found in this folder.")