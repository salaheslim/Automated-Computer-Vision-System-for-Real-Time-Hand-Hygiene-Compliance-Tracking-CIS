import subprocess
import json

helper = subprocess.Popen(
    [r'env_xgboost\Scripts\python.exe', 'landmark_helper_with_frame.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True
)

print("Reading from helper...")
for i in range(3):
    line = helper.stdout.readline().strip()
    if line:
        data = json.loads(line)
        print(f"Line {i+1}: frame size={len(data['frame'])} left={data['left'][:3]}")
    else:
        print(f"Line {i+1}: EMPTY - helper not sending data")

helper.terminate()
print("Done")
q