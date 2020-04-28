import pymongo
import config

def connect_to_database():
    global db
    mongo_name = config.mongo["user"]
    mongo_password = config.mongo["passwd"]

    # Connect to the Mongo Database
    client = pymongo.MongoClient()
    db = client[config.mongo["db"]]
    db.authenticate(name=mongo_name, password=mongo_password)
    return db

db = connect_to_database()

def insert_data(collection_name,data_dictionary):
	result = db[collection_name].insert_one(data_dictionary)
	return result

def clean(text):
	text = text.replace("\r","")
	return text 

def insert_issue(issue,collection_name,ghtorrent_collection,label=''):
	# Issue must be in form user/repo/issue
	user,repo,issue_number = issue.split("/")
	fields_to_take = ["title","repo","owner","number"]

	results = db[ghtorrent_collection].find({'repo':repo,'owner':user,'number':int(issue_number)})
	for result in results:
		d = {}
		for field in fields_to_take:
			d[field] = result[field]

		d['text'] = clean(result['body'])

		if label:
			d['toxic'] = label

		print(d)

def insert_all_comments(issue,collection_name,ghtorrent_collection,label={}):
	user,repo,issue_number = issue.split("/")

insert_issue("storybooks/storybook/4304","naveen_temp","issues")
