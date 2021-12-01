import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
def Prepare(eval_frames,from_path,save_path,num_eval):
    tstart = time.time()
    feats = []
    read_file = Path(from_path)
    files = []
    used_speaker = []
    with open(read_file) as listfile:
        while True:
            line  = listfile.readline()
            if(not line):
                break
            data = line.split()

            data_1_class = Path(data[1]).parent.stem
            data_2_class = Path(data[2]).parent.stem
            if data_1_class not in used_speaker:
                used_speaker.append(data_1_class)
                files.append(data[1])
            if data_2_class not in used_speaker:
                used_speaker.append(data_2_class)
                files.append(data[2])
            setfiles = list(set(files))
            setfiles.sort()
            #save all features to file
            for idx, f in enumerate(tqdm(setfiles)):
                inp1 = torch.FloatTensor(
                    loadWAV(f,eval_frames,evalmode=True,num_eval=num_eval)
                ).to(device)
                feat



