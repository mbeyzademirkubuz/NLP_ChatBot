from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import sys
import io
import locale


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Türkçe karakterleri anlaması için
locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')

# Open JSON file with UTF-8 encoding
with open('chatbot_env/intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

@app.route('/chat', methods=['POST'])
def chat():
    sentence = request.json.get('message')
    if not sentence:
        return jsonify({"response": "I do not understand..."})

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent['responses'])})
    else:
        return jsonify({"response": "Üzgünüm. Ne dediğini anlayamadım..."})

if __name__ == "__main__":
    app.run(debug=True)
