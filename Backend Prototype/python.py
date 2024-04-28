from flask import Flask, render_template, request, jsonify
import random
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



app = Flask(__name__)

customerID = ""
email = ""


model = load_model("D:\\invergy\\Backend Prototype\\chatbot_model.h5")

data = pickle.load(open('D:\\invergy\\Backend Prototype\\training_data.pkl', 'rb'))
words = data['words']
classes = data['classes']

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = 'None'
    return result

def email_info(sender_email, sender_password, receiver_email, subject, message):
    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)  # Replace 'smtp.example.com' with your SMTP server address
    smtp_server.login(sender_email, sender_password)

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach message to email
    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    smtp_server.send_message(msg)

    # Close the SMTP server
    smtp_server.quit()

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/getInfo', methods=['POST'])
def get_info():
    global customerID, email
    customerID = request.form['id']
    print(customerID)
    return customerID
    

@app.route('/chat', methods=['POST'])
def chat():
    with open('D:\\invergy\\Backend Prototype\\database.json') as file:
        intents = json.load(file)
    
    user_input = request.form['text']
    user_input.lower()
    ints = predict_class(user_input)
    print(ints)
    int_class = ints[0]['intent']
    res = get_response(ints, intents)

    
    if user_input.lower() == 'quit':
        return jsonify({"response": "Goodbye!"})
    elif res != 'None':
        if int_class == "product_not_working_properly" or int_class == 'product_damaged' or int_class == "product_replace_request":
            email_info("drm281208@gmail.com", "bkfr qzja isvo pmqj", "maloobrothers28@gmail.com", ("Consumer Complaint" + " " + int_class), ("The client with customer ID " + customerID + " , has the following complaint: \n" + str(user_input)))
            print('Successfully send an email!!')
        return jsonify({"response": res})
    
    else:
        return jsonify({
            "response": "Sorry, I am not able to understand your query."
        })
    
    

if __name__ == '__main__':
    app.run(debug = True)


