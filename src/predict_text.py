import pickle
import suite

model = pickle.loads(open("pretrained_model.p","rb").read())

def predict_text(text):
	return suite.get_prediction(text,model)


