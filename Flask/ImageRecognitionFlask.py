#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data


#system level operations (like loading files)
import sys 
#for reading operating system data
import os
sys.path.append(os.path.abspath("./model"))
#from LoadImage import * 


app = Flask(__name__)

#@app.route('/index',methods=['POST'])



#get Image into format NN can run
def processImage(image, target_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(image(64,64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image
print("load images")


@app.route("/about")
def about():
	return render_template("about.html")


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


"""

def process_image(image_data):
    imagestring = re.search(r'base64,(.*',image_data1).group(1)
    with open('image.jpg', 'wb') as out:
        output.write(imagestring.decode('base64'))




@app.route("/predict", methods=["POST"])
def predict():

	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	
	img = imread('output.jpg')
	#make it the right size
	img = imresize(x,(64,64))

	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,64,64,3)
	#print "debug2"
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#print "debug3"
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
"""



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)