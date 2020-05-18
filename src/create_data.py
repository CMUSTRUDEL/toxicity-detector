import pymongo
import config
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textblob
import perspective
import TextParser

import sys
sys.path.insert(0, "politeness3")
import politeness3.model

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
	text = TextParser.remove_code(text)
	text = TextParser.remove_inline_code(text)
	text = TextParser.remove_newline(text)
	text = TextParser.replace_mention(text)
	text = TextParser.sub_PlusOne(text)
	text = TextParser.remove_comments(text)
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

		print("Inserting {}".format(d))
		#insert_data(collection_name,d)

def insert_all_comments(issue,collection_name,ghtorrent_collection,label=[]):
	user,repo,issue_number = issue.split("/")
	field_names = {'owner':'owner','repo':'repo','issue_id':'issue_id'}

	all_dicts = []

	sid = SentimentIntensityAnalyzer()

	results = db[ghtorrent_collection].find({'repo':repo,'owner':user,'issue_id':int(issue_number)})
	for result in results:
		d = {}

		for field in field_names:
			d[field_names[field]] = result[field]
		d['text'] = clean(result['body'])

		comment_id = result['id']
		d['comment_id'] = comment_id
		if comment_id in label:
			d['toxic'] = label[comment_id]

		d['_id'] = "{}/{}/{}/{}".format(repo,user,issue_number,comment_id)
		d['nltk_score'] = sid.polarity_scores(d['text'])['compound']

		textblob_scores = textblob.TextBlob(d['text'])
		d['polarity'] = textblob_scores.polarity
		d['subjectivity'] = textblob_scores.subjectivity

		tokenized = nltk.sent_tokenize(d['text'])
		d['stanford_polite'] = politeness3.model.score({'sentences':tokenized,'text':d['text']})['polite']
		d['perspective_score'] = perspective.get_perspective_score(d['text'])

		print("Inserting {}".format(d))

		#insert_data(collection_name,d)

insert_all_comments("rstudio/shiny/1595","naveen_temp_comments","issue_comments")
