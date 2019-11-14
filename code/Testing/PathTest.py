import os
import platform


contents = ""
if platform.system() == "Windows":
    try:
        currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\Testing", "")
        f = open(currentDir + "\\code\\Testing\\RandText.txt", "r")
        contents = f.read()
    except:
        print("FAIL")
else:
    try:
        currentDir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").replace("code/Testing", "")
        f = open(currentDir + "/code/Testing/RandText.txt", "r")
        contents = f.read()
    except:
        print("FAIL")
if contents == "Alpha Bravo Charlie":
    print("SUCCES")
