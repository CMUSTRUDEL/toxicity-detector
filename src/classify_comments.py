# this will classify all issues and comments as toxic
# in mongodb, starting with the newest
import logging
# logging.basicConfig(filename='toxicity_issues.log',level=logging.INFO)
# logging.basicConfig(level=logging.INFO)

import config

VERSION = "v1"
TABLE_PREFIX = "christian_toxic_"

logging.info("loading")
import pickle
import pymongo
import time


logging.info("loading model")
model = pickle.load(open("pretrained_model.p","rb"))
import suite


logging.info("connecting to database")
def connect_to_database():
	mongo_name = config.mongo["user"]
	mongo_password = config.mongo["passwd"]

	# Connect to the Mongo Database
	client = pymongo.MongoClient()
	db = client[config.mongo["db"]]
	db.authenticate(name=mongo_name, password=mongo_password)
	return db
db = connect_to_database()

logging.info("starting")





def get_next_date(table): # updated_at date as string
	r = db[TABLE_PREFIX+table].find_one(
			filter= {"toxicity."+VERSION: {"$exists": 0}}, 
			sort= [("updated_at", -1)]
		 )
	return r["updated_at"]

def claim_next(table): # [id, updated_at, time]
	start = time.time()
	r = db[TABLE_PREFIX+table].find_one_and_update(
			filter= {"toxicity."+VERSION: {"$exists": 0}}, 
			sort= [("updated_at", -1)], 
			update= {"$set":{"toxicity."+VERSION+".in_progress": 1 }} 
		 )
	return [r["_id"], r["updated_at"], time.time() - start]

def get_text(table, id): # [text, time]
	start = time.time()
	i = db[table].find_one({"_id": id}, {"title":1, "body":1})
	text = ""
	if "title" in i:
		text += i["title"] + ": "
	if "body" in i:
		text += i["body"] 
	# print(text)
	return [text, time.time() - start]


def update_db(table, id, new_data):
	start = time.time()
	db[TABLE_PREFIX+table].update({"_id": id},{ "$set": new_data })
	return time.time() - start

def compute_prediction_report(text):
	start = time.time()
	# score the issue's text
	score = suite.score_toxicity(text, model)
	result = { 
			"score": score[0].item(), 
			"orig" : {"score": score[0].item(), "persp": score[1], "polite": score[2]},			
			}

	# if toxic, look at alternatives
	if score[0]==1:
		alt_text = suite.clean_text(text)
		if len(alt_text)==0:
			print(" == found toxic issue, no alternatives")
		else:
			print(" == found toxic issue, exploring "+str(len(alt_text))+" alternatives")
			isToxic = True
			for a in alt_text:
				if isToxic:
					score = suite.score_toxicity(text, model)
					if score[0] == 0:
						print(" === found nontoxic alternative")
						isToxic=False
						result["score"]=0
						result["alt"]={"text":a,"score": score[0].item(), "persp": score[1], "polite": score[2]}
			if not isToxic:
				result["alt_tried"]=len(alt_text)
	return [result, time.time() - start]

def process_one_item(table):
	# grab the most recent issue to process
	[issue_id, d, t1] = claim_next(table)
	print(table, issue_id, d)

	# get the text
	[text, t2] = get_text(table, issue_id)

	# score the text
	[score_report, t3] = compute_prediction_report(text)
	result = {"toxicity."+VERSION: score_report}

	# write results to db
	t4=update_db(table,issue_id,result)
	# print("db time", t1, t2, t4, "scoring time", t3)

def process_100_items(table):
	for x in range(0, 99):
		process_one_item(table)

while True:
	next_i = get_next_date("issues")
	next_ic = get_next_date("issue_comments")
	if (next_i > next_ic):
		process_100_items("issues")
	else:
		process_100_items("issue_comments")
	