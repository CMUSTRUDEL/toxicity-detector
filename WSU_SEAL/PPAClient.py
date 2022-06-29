import requests
import  json

api_key = 'YOUR_KEY_HERE' # please do not forget to add your API key here
url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
       '?key=' + api_key
       )


def get_api_response(data_dict):

    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content)
    return  response_dict

def get_perspective_api_score(text):
    data_dict = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}}
    response_dict = get_api_response(data_dict)
    #print(response_dict)
    toxicity_score =json.dumps(response_dict['attributeScores']['TOXICITY']['summaryScore']['value'])
    return toxicity_score


#value =get_perspective_api_score("I am fine.")
#print(value)