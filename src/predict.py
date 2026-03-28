import pickle

model = pickle.load(open("models/model.pkl", "rb"))

def predict(data):
    prediction = model.predict([data])[0]
    probs = model.predict_proba([data])[0]

    malignant = probs[0]
    benign = probs[1]

    return prediction, malignant, benign