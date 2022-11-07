end = '\033[0m'

def printc(text): # Cyan
    start = '\033[96m'
    print(f"{start}{text}{end}")

def printo(text): # OK
    start = '\033[92m'
    print(f"{start}{text}{end}")

def printw(text): # Warning
    start = '\033[93m'
    print(f"{start}{text}{end}")

def printe(text): # Error
    start = '\033[91m'
    print(f"{start}{text}{end}")