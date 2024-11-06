import re
import os
import argparse
import urllib.request
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

backends = [
    # "ibmq_lima", 
    # "ibmq_belem", 
    # "ibmq_quito", 
    # "ibmq_manila", 
    # "ibmq_jakarta", 
    # "ibm_oslo", 
    # "ibm_nairobi", 
    # "ibm_lagos", 
    # "ibm_perth",
    "ibm_sherbrooke",
    "ibm_kyiv",
    "ibm_brisbane",
    "ibm_fez",
    "ibm_kawasaki",
    "ibm_nazca",
    "ibm_torino",
    "ibm_quebec",
    "ibm_rensselaer",
    "ibm_strasbourg",
    "ibm_brussels"
]

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def download_file(backend, user, pw):
    # Create target folder if it does not exist
    target_folder = f"C:/Users/Ivo/Documents/IBM Calibration Data/{backend.split('_')[-1]}/"
    make_all_dirs(target_folder)
    # Create a new instance of the Chrome driver
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = 'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe'
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": target_folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    driver = webdriver.Chrome('C:/Users/Ivo/Downloads/chromedriver_win32/chromedriver.exe', options=chrome_options)

    # Navigate to the login page
    driver.get("https://login.ibm.com/authsvc/mtfim/sps/authsvc?PolicyId=urn:ibm:security:authentication:asf:basicldapuser&Target=https%3A%2F%2Flogin.ibm.com%2Fauthsvc%2F")
    driver.implicitly_wait(500)

    # Find the username and password fields and enter your credentials
    username_field = driver.find_element_by_id("username")
    username_field.send_keys(user)
    username_field.send_keys(Keys.RETURN)
    driver.implicitly_wait(1000)
    password_field = driver.find_element_by_id("password")
    password_field.send_keys(pw)
    # Submit the login form
    password_field.send_keys(Keys.RETURN)


    # Wait for the login to complete
    driver.implicitly_wait(1000)

    driver.get(f"https://auth.quantum-computing.ibm.com/auth/idaas?redirectTo=https%3A%2F%2Fquantum-computing.ibm.com%2Fservices%2Fresources%3Ftab%3Dyours%26system%3D{backend}")
    first_button = driver.find_element_by_xpath('//*[@id="__layout"]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/main/div/div[1]/div[1]/div[2]/div/div[3]/div[2]/div/div/div/div/div/div/ul/li[3]/div/div/div[1]/div[2]/div/div[1]/button')
    first_button.click()
    time.sleep(1)
    download_button = driver.find_element_by_xpath("//button[contains(text(), 'Download all calibration data (.csv)')]")
    # js_function_call = download_button.get_attribute('onclick')
    # url_match = re.search(r"'(https?://\S+)'", js_function_call)
    # download_url = url_match.group(1)
    time.sleep(1)
    download_button.click()
    # Wait until download is finished
    time.sleep(5)
    # wait.until(EC.invisibility_of_element_located((By.XPATH, "//button[contains(text(), 'Download all calibration data (.csv)')]")))

    # Close the browser window
    driver.quit()
    print(f"File for {backend} downloaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

# def download_file(url, file_path):
#     urllib.request.urlretrieve(url, file_path)
#     print(f"File downloaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-b", "--backend", default="manila", type=str,
    #     help="Backend for which calibration data is downloaded."
    # )
    parser.add_argument(
        "-t", "--sleep_time", default=1200, type=int,
        help="Time between calibration csv's download in microsec."
    )
    parser.add_argument(
        "-u", "--username", default="ivo.mihov@protonmail.com", type=str,
        help="Username for the IBM login."
    )
    parser.add_argument(
        "-p", "--pw", default="ztR3=0wP", type=str,
        help="Password for the IBM login."
    )
    args = parser.parse_args()
    sleep_time = args.sleep_time
    # backend = args.backend
    user = args.username
    pw = args.pw
    # backend = "ibm_" + backend \
    #     if backend in ["perth", "lagos", "nairobi", "oslo"] \
    #         else "ibmq_" + backend
    while True:
        for backend in backends:
            download_file(backend, user, pw)
        time.sleep(sleep_time)