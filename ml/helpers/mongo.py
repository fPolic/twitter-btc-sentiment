import motor.motor_asyncio

# TODO: handle trough config file or env (Docker)
DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'

# global instance
DB = None

#  returns mongo instance singleton


def get_db_instance(db_name=DATABASE_NAME):
    global DB
    if DB is None:
        client = motor.motor_asyncio.AsyncIOMotorClient(mongoURL)
        DB = client[DATABASE_NAME]
    return DB
