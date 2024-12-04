import os

if os.path.exists("./.env"):
    print(".env file found.")
else:
    print(".env file missing.")
