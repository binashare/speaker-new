import requests
files = {'file1': open('C:/DoAnToNghiep/Speaker_New/speaker_server-master_2/app/core/data/negative/00001.wav', "rb"), 'file2': open('C:/DoAnToNghiep/Speaker_New/speaker_server-master_2/app/core/data/negative/00003.wav', "rb")}
resp = requests.post('http://localhost:5000/predict', files=files)
print(resp.text)