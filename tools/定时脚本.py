import schedule
import os
import subprocess
import time

id = 0
def job1():
    global id
    # a = subprocess.Popen(r"conda activate AI && python test.py" , shell=True,cwd="F:\下载\数据集传输")
    a = subprocess.Popen(r"conda activate AI && python faceswap.py train -A output\face3 -B output\face1 -m models", shell=True,cwd=r"F:\facewap")
    id = a.pid
def job2():
    os.system(f"taskkill /pid {id}  -t -f")

schedule.every().day.at("17:43").do(job1)
schedule.every().day.at("17:45").do(job2)

while True:
    schedule.run_pending()
    time.sleep(1)