import os
import json
import time
import argparse
import requests
from datetime import datetime

import numpy as np

# backends = [
#     "ibmq_lima", 
#     "ibmq_belem", 
#     "ibmq_quito", 
#     "ibmq_manila", 
#     "ibmq_jakarta", 
#     "ibm_oslo", 
#     "ibm_nairobi", 
#     "ibm_lagos", 
#     "ibm_perth"
# ]

backends = [
    "ibm_kyoto",
    "ibm_brisbane"
]
def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def download_file(backend, access_token):
    url=f'https://api.quantum-computing.ibm.com/api/Backends/{backend}/properties'
    # BE CAREFUL TO CHANGE THE NEXT LINE TO THE DESIRED FOLDER (or you will have user Ivo created ;))
    
    home_path = os.path.expanduser(os.getenv('USERPROFILE'))

    save_dir = f"{home_path}/Documents/IBM Calibration Data/{backend.split('_')[-1]}/"
    make_all_dirs(save_dir)
    headers = {
        'authority': 'api.quantum-computing.ibm.com',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9,bg-BG;q=0.8,bg;q=0.7',
        'dnt': '1',
        'origin': 'https://quantum-computing.ibm.com',
        'referer': 'https://quantum-computing.ibm.com/',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        'x-access-token': access_token
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_path = os.path.join(
            save_dir, 
            f"calibration_data_{backend}_{datetime.now().strftime('%Y-%m-%dT%H%M%S')}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(response.json(), f)
        print(f"Successfully downloaded data for {backend}!")
    else:
        print('Request failed with status code', response.status_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--sleep_time", default=1200, type=int,
        help="Time between calibration csv's download in microsec."
    )
    parser.add_argument(
        "-at", "--access_token", 
        # turns out the access token is not needed
        default="", 
        type=str, help="Access token for the IBM Quantum website."
    )
    args = parser.parse_args()
    sleep_time = args.sleep_time
    access_token = args.access_token
    while True:
        for backend in backends:
            download_file(backend, access_token)
        print(f"Sleeping for {np.round(sleep_time / 60)} minutes.")
        time.sleep(sleep_time)