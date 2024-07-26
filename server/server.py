# Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings
import os.path

# Feel free to use this in your individual or commercial projects BUT make sure to reference me as: Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
# It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

# Use at your own risk
# Use under the Apache License 2.0

# Example of a Python UDP server

import UdpComms as U
import time
import subprocess
import json

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

i = 0

MODE = "video"
CALC_SMPL = "--calc_smpl"
RENDER_MESH = "--render_mesh"
FRAME_RATE = 30
SAVE_VIDEO = "--save_video"
output_path = "vid_demo/output/test.mp4"

while True:
    data = sock.ReadReceivedData()
    print(f"data: {data}")
    if data is None:
        continue
    data_dict = json.loads(data)
    if "video" in data_dict:
        video_path = data_dict['video']

        # subprocess.run(
        #     ['python', './romp/main.py', f"--mode={MODE}", CALC_SMPL, RENDER_MESH, f"--frame_rate={FRAME_RATE}",
        #      f"-i={video_path}", f"-o={output_path}", SAVE_VIDEO])
        # print("done render vid")
        result = {
            'video': os.path.abspath(output_path)
        }

        time.sleep(5)
        print(json.dumps(result))

        # Send this string to other application
        sock.SendData(json.dumps(result))
    # i += 1
    #
    # data = sock.ReadReceivedData() # read data
    #
    # if data != None: # if NEW data has been received since last ReadReceivedData function call
    #     print(data) # print new received data
    #
    time.sleep(1)
