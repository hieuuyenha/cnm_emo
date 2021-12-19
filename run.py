from flask import Flask, render_template, request,Response
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
	return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	image = request.files['select_file']

	image.save('static/file.jpg')

	image = cv2.imread('static/file.jpg')

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	
	faces = cascade.detectMultiScale(gray, 1.1, 3)

	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

		cropped = image[y:y+h, x:x+w]


	cv2.imwrite('static/after.jpg', image)
	try:
		cv2.imwrite('static/cropped.jpg', cropped)

	except:
		pass



	try:
		img = cv2.imread('static/cropped.jpg', 0)

	except:
		img = cv2.imread('static/file.jpg', 0)

	img = cv2.resize(img, (48,48))
	img = img/255

	img = img.reshape(1,48,48,1)

	model = load_model('mdfinal.h5')

	pred = model.predict(img)


	label_map = ['tức giận','bình thường' , 'sợ hãi', 'vui vẻ', 'buồn', 'bất ngờ']
	pred = np.argmax(pred)
	final_pred = label_map[pred]


	return render_template('predict.html', data=final_pred)

camera = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
model = load_model('mdfinal.h5')

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
            faces_detected = face_haar_cascade.detectMultiScale(gray_img)  
            
        
            for (x,y,w,h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img_pixels = image.img_to_array(roi_gray)  
                img_pixels = np.expand_dims(img_pixels, axis = 0)  
                img_pixels /= 255  
                
        
                print(img_pixels.shape)
                
                predictions = model.predict(img_pixels)  
        
                #find max indexed array  
                
                max_index = np.argmax(predictions[0])  
        
                emotions = ['tuc gian','binh thuong' , 'so hai', 'vui ve', 'buon', 'bat ngo']  
                predicted_emotion = emotions[max_index]  
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
            resized_img = cv2.resize(frame, (1000, 700))  
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/cam')
def cam():
	 return render_template('index.html')

@app.route('/about')
def about():
	 return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	app.run(debug=True)