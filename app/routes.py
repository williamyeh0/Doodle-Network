from flask import render_template, request, jsonify, g
from app import app
from PIL import Image
import base64
import io
import numpy as np
from models.neural_net import NeuralNetwork
import logging
import uuid
import time

logging.basicConfig(level=logging.INFO)

@app.before_request
def before_request_func():
    execution_id = uuid.uuid4()
    g.start_time = time.time()
    g.execution_id = execution_id
    app.logger.info(f"Received request: {g.execution_id, request.url, request.method}")

@app.route('/')
def index():
    user = {'username': 'Miguel'}
    return render_template('paint.html', title='Home', user=user)
@app.route('/paint', methods =['GET', 'POST'])
def predict():
    # app.logger.info(f"Received request: {request.url, request.method}") #logging
    np.set_printoptions(suppress=True)
    # request --> base64
    base64_image = request.json['image']
    
    # URL prefix
    if 'data:image/png;base64,' in base64_image:
        base64_image = base64_image.split('data:image/png;base64,')[1]
    
    #convert image to 784 pixel np array neural net input
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('L') #grayscale
    image = image.resize((28, 28)) #scale down
    # image.show()

    #subtract mean image
    doodle = np.array(image, dtype='float64').flatten()
    mean_images = np.load('data/MNIST/raw/mean_images.npz') #root is repo
    doodle -= mean_images['digits'] #for digits
    
    # np.save('test_doodle.npy', doodle)

    #load network
    params = np.load('models/model_params.npz')
    hyperparams = np.load('models/model_hyperANDparams.npz', allow_pickle=True)
    #make sure hidden_sizes is a list
    input_size, hidden_sizes, output_size, num_layers = hyperparams['input_size'], list(hyperparams['hidden_sizes']), hyperparams['output_size'], hyperparams['num_layers']
    
    net = NeuralNetwork(input_size, hidden_sizes, output_size, num_layers, opt="SGD") #opt doesn't matter
    net.params = params #please 

    #predict
    prediction = np.round(net.forward(doodle), 6) #probabilities
    sorted_classes = np.argsort(prediction)[::-1] #classes
    prediction_pairs = [ (str(num), str(float(prediction[num]))) for num in sorted_classes] #strings
    # print("prediction_pairs: ", prediction_pairs) 

    # return prediction, this is just shape for now
    return jsonify(prediction_pairs)