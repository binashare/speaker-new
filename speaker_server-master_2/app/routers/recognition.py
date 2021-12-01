import io
import os
import shutil
from fastapi import APIRouter, File, UploadFile,Form, Body
from fastapi.encoders import jsonable_encoder
from typing import List
import pandas as pd
from core.voice_utils import compare_similarity
from core.voice_utils import extra_feature
from core.voice_utils import check
import torch.nn.functional as F
from pydantic import BaseModel, Field
import torch
import json
import numpy as np
from server.database import FindAll
router = APIRouter(
    prefix="/recognition",
    tags=["recognition"],
)



class ItemList(BaseModel):
    feature: List[float]
file_feature = "C:/DoAnToNghiep/Speaker_New/speaker_server-master_2/app/list_feature.csv"
temporary_Folder = "C:/DoAnToNghiep/Speaker_New/speaker_server-master_2/app/temporary"
@router.get("/get_feature")
async def get_feature(audio: str):
    try:
        print("da vao day")
        name = audio.split('\\')[2]
        temporary_file = temporary_Folder + '/' + name + '.pt'
        print(name)
        feature_1 = extra_feature(audio=audio)
        # feature_1 = feature_1.detach().cpu().numpy().tolist()
        #feature_1 = np.asarray(feature_1.detach().cpu().numpy().tolist())
        torch.save(feature_1, temporary_file)
        # feature_1 = str(feature_1)
        list_feature = pd.read_csv(file_feature)
        list_feature = list_feature.append(pd.DataFrame({
            "label": [name],
            "path": [temporary_file]
        }))
        list_feature.to_csv(file_feature,index=False)
        data = {"feature":feature_1}
        return data
    except Exception as err:
        print(err)
        return {'error': 'error during get feature'}
@router.get("/extract_feature")
async def extract_feature(audio: str):
    try:
        print(audio)
        feature_1 = extra_feature(audio=audio)
        result = feature_1.detach().cpu().numpy().tolist()
        return result
    except Exception as err:
        print(err)
        return {'error': 'error during get feature'}
@router.post("/check_audio")
async def check_audio(audio:UploadFile = File(...)):
    print(audio.filename)
    result = check(audio.file)
    print(result)
    data = {'result':result}
    print(data)
    return data
@router.get("/reset")
def resetFolder():
    try:
        if os.path.exists(temporary_Folder):
            shutil.rmtree(temporary_Folder)
        if not os.path.exists(temporary_Folder):
            os.makedirs(temporary_Folder)
        database_join = pd.read_csv(file_feature)
        database_join = database_join[database_join['label'] == "/"]
        database_join.to_csv(file_feature, index=False)
        data = {'result': True}
        return data
    except Exception as err:
        return {'error': 'error during removing'}
@router.post("/get_Top5")
def get_Top5(audio:UploadFile = File(...)):
    try:
        result = {}
        feature_1 = extra_feature(audio=audio.file)
        feature_1 = feature_1.detach().cpu().numpy()
        # list_feature = pd.read_csv(file_feature)
        cursor = FindAll()

        # for tmp in list_feature.values:
        for tmp in cursor:

            # feature_2 = torch.load(tmp[1])
            feature_2 = tmp['Feature']
            # print(feature_2)
            score = F.cosine_similarity(torch.from_numpy(np.array(feature_2)),torch.from_numpy(feature_1), eps=1e-08)
            score = score.cpu().numpy()
            score = np.mean(score)
            if(score > 0.40):
                result[tmp['CMND']] = score
                print(tmp['TenDT'], score)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        fout = "C:/DoAnToNghiep/Speaker_New/NCKH/text.txt"
        fo = io.open(fout, "w",encoding="utf-8")
        for index, item in enumerate(result):
            if(index < 5):
                fo.write('%s:%s\n' % (item[0], item[1]))
        fo.close()
        data = {'result': True}
        return data
    except Exception as err:
        print(err)
        return {'error': 'error during get feature'}