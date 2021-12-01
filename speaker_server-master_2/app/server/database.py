import motor.motor_asyncio
from bson.objectid import ObjectId

MONGO_DETAILS = "mongodb://localhost:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
user_collection = client['local'].get_collection("Infor_Speaker")

def user_helper(user):
    return {
        "id": str(user["_id"]),
        "TenDT": user["TenDT"],
        "CMND": user["CMND"],
        "Phone": user["Phone"],
        "QuocTich": user["QuocTich"],
        "QueQuan": user["QueQuan"],
        "NgaySinh": user["NgaySinh"],
        "GioiTinh": user["GioiTinh"],
        "pathFile": user["pathFile"],
    }

# Add a new user into to the database
async def add_user(user_data):
    CMND = await user_collection.find_one({"CMND": user_data['CMND']})
    if CMND is None:
        user = await user_collection.insert_one(user_data)
        # print('ggg',user.inserted_id)
        new_user = await user_collection.find_one({"_id": user.inserted_id})
        return user_helper(new_user)
    else: 
        return False

async def login(user_data):
    result = await user_collection.find_one({"number_phone": user_data['number_phone']})
    return result

async def update_user(user_data):
    try:
        user_update = await user_collection.update_one({"_id": user_data['_id']}, {"$set": user_data})
        return True
    except:
        return False
async def FindUserbyCMND(user_data):
    result = await user_collection.find_one({"CMND": user_data['CMND']})
    return result
def FindAll():
    import pymongo
    client = pymongo.MongoClient('mongodb://localhost:27017')
    database = client['local']
    collect = database.get_collection("Infor_Speaker")
    cursor = collect.find()
    return cursor






    