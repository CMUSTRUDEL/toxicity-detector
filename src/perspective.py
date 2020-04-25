import time

import config

API_KEY = config.perspective_api_key
# 
'''
code from the perspective wrapper: https://github.com/Conway/perspective
'''

import time
import json
import requests
import warnings
import markdown
import logging
from retrying import retry

#from get_data import *
try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser
def validate_language(language):
    # ISO 639-1 code validation
    # language source: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
    codes = ["ab", "aa", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay",
             "az", "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
             "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", "da", "de",
             "dv", "dz", "ee", "el", "en", "eo", "es", "et", "eu", "fa", "ff",
             "fi", "fj", "fo", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "gv",
             "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz", "ia", "id",
             "ie", "ig", "ii", "ik", "io", "is", "it", "iu", "ja", "jv", "ka",
             "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku",
             "kv", "kw", "ky", "la", "lb", "lg", "li", "ln", "lo", "lt", "lu",
             "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
             "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
             "oc", "oj", "om", "or", "os", "pa", "pi", "ps", "pt", "qu", "rm",
             "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk",
             "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
             "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts",
             "tt", "tw", "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo", "wa",
             "wo", "xh", "yi", "yo", "za", "zh", "zu"]
    return language.lower() in codes


def remove_html(text, md=False):
    if md:
        text = markdown.markdown(text)
    # credit: stackoverflow
    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs= True
            self.fed = []
        def handle_data(self, d):
            self.fed.append(d)
        def get_data(self):
            return ''.join(self.fed)

    s = MLStripper()
    s.feed(text)
    return s.get_data()
# allowed test types
allowed = ["TOXICITY",
           "SEVERE_TOXICITY",
           "TOXICITY_FAST",
           "ATTACK_ON_AUTHOR",
           "ATTACK_ON_COMMENTER",
           "INCOHERENT",
           "INFLAMMATORY",
           "OBSCENE",
           "OFF_TOPIC",
           "UNSUBSTANTIAL",
           "LIKELY_TO_REJECT"]

class Perspective(object):

    base_url = "https://commentanalyzer.googleapis.com/v1alpha1"

    def __init__(self, key):
        self.key = key

    def score(self, text, tests=["TOXICITY"], context=None, languages=["en"], do_not_store=False, token=None, text_type=None):
        # data validation
        # make sure it's a valid test
        # TODO: see if an endpoint that has valid types exists
        if isinstance(tests, str):
            tests = [tests]
        if not isinstance(tests, (list, dict)) or tests is None:
            raise ValueError("Invalid list/dictionary provided for tests")
        if isinstance(tests, list):
            new_data = {}
            for test in tests:
                new_data[test] = {}
            tests = new_data
        if text_type:
            if text_type.lower() == "html":
                text = remove_html(text)
            elif text_type.lower() == "md":
                text = remove_html(text, md=True)
            else:
                raise ValueError("{0} is not a valid text_type. Valid options are 'html' or 'md'".format(str(text_type)))

        for test in tests.keys():
            if test not in allowed:
                warnings.warn("{0} might not be accepted as a valid test.".format(str(test)))
            for key in tests[test].keys():
                if key not in ["scoreType", "scoreThreshhold"]:
                    raise ValueError("{0} is not a valid sub-property for {1}".format(key, test))

        # The API will only grade text less than 3k characters long
        if len(text) > 3000:
            # TODO: allow disassembly/reassembly of >3000char comments
            warnings.warn("Perspective only allows 3000 character strings. Only the first 3000 characters will be sent for processing")
            text = text[:3000]
        new_langs = []
        if languages:
            for language in languages:
                language = language.lower()
                if validate_language(language):
                    new_langs.append(language)

        # packaging data
        url = Perspective.base_url + "/comments:analyze"
        querystring = {"key": self.key}
        payload_data = {"comment": {"text": text}, "requestedAttributes": {}}
        for test in tests.keys():
            payload_data["requestedAttributes"][test] = tests[test]
        if new_langs != None:
            payload_data["languages"] = new_langs
        if do_not_store:
            payload_data["doNotStore"] = do_not_store
        payload = json.dumps(payload_data)
        headers = {'content-type': "application/json"}
        # t=time.time()
        response = requests.post(url,
                            data=payload,
                            headers=headers,
                            params=querystring,
                            timeout=10)
        # print("persp time", time.time()-t)
        data = response.json()
        if "error" in data.keys():
            raise PerspectiveAPIException(data["error"]["message"])
        c = Comment(text, [], token)
        base = data["attributeScores"]
        for test in tests.keys():
            score = base[test]["summaryScore"]["value"]
            score_type = base[test]["summaryScore"]["type"]
            a = Attribute(test, [], score, score_type)
            for span in base[test]["spanScores"]:
                beginning = span["begin"]
                end = span["end"]
                score = span["score"]["value"]
                score_type = span["score"]["type"]
                s = Span(beginning, end, score, score_type, c)
                a.spans.append(s)
            c.attributes.append(a)
        # print("perspective call: ",text,base)
        return c

class Comment(object):
    def __init__(self, text, attributes, token):
        self.text = text
        self.attributes = attributes
        self.token = token

    def __getitem__(self, key):
        if key.upper() not in allowed:
            raise ValueError("value {0} does not exist".format(key))
        for attr in self.attributes:
            if attr.name.lower() == key.lower():
                return attr
        raise ValueError("value {0} not found".format(key))

    def __str__(self):
        return self.text

    def __repr__(self):
        count = 0
        num = 0
        for attr in self.attributes:
            count += attr.score
            num += 1
        return "<({0}) {1}>".format(str(count/num), self.text)

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self):
        return len(self.text)

class Attribute(object):
    def __init__(self, name, spans, score, score_type):
        self.name = name
        self.spans = spans
        self.score = score
        self.score_type = score_type

    def __getitem__(self, index):
        return self.spans[index]

    def __iter__(self):
        return iter(self.spans)

class Span(object):
    def __init__(self, begin, end, score, score_type, comment):
        self.begin = begin
        self.end = end
        self.score = score
        self.score_type = score_type
        self.comment = comment

    def __str__(self):
        return self.comment.text[self.begin:self.end]

    def __repr__(self):
        return "<({0}) {1}>".format(self.score, self.comment.text[self.begin:self.end])

class PerspectiveAPIException(Exception):
    pass
p = Perspective(API_KEY)
import nltk
words = set(nltk.corpus.words.words())


def remove_non_english(text):
    text = text.replace('"','').replace("'",'')
    text = (w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words and w.isalpha())

    return " ".join(text)

#retry, up to 1h delay, never give up waiting
@retry(wait_exponential_multiplier=10000, wait_exponential_max=3600000)
def get_perspective_score(text):
    # time.sleep(0.02)

    if len(text.strip()) == 0:
        return 0

    try:
        comment = p.score(text,tests=["TOXICITY"])
        score = comment["TOXICITY"].score
        return score
    except Exception as e:
        print(e)
        raise e

