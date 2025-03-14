from django.shortcuts import render, HttpResponse
from django.conf import settings
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from django.core.files.storage import FileSystemStorage
import os
# adding for camera access
# from django.http import JsonResponse
# from django.core.files.storage import FileSystemStorage

from django.http import StreamingHttpResponse
import cv2


# from .deepfake_detection import process_video  # Assuming you have a deepfake model processing script


# Create your views here.

def index(request):
    context = {
        "var1": "aarti",
        "var2": "jha"
    }
    return render(request, 'deepfake/index.html', {'STATIC_URL': settings.STATIC_URL})


def about(request):
    return render(request, 'deepfake/about.html')
    # return HttpResponse("this is about page")


def services(request):
    return render(request, 'deepfake/services.html')
    # return HttpResponse("this is services page")


def contact(request):
    return render(request, 'deepfake/contact.html')
    # return HttpResponse("this is contact page")


# def index(request):
#     return render(request, 'detection/index.html')

def real_video(request):
    return render(request, 'deepfake/real_video.html')


def upload_video(request):
    return render(request, 'deepfake/upload_video.html')


def upload_photo(request):
    return render(request, 'deepfake/upload_photo.html')

    # Adding its code


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

    # UPLOAD VEDIO


def process_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Define path to save the file
        upload_path = os.path.join("media", "video", "video.mp4")

        # Save the uploaded video with a fixed filename
        fs = FileSystemStorage(location=os.path.dirname(upload_path))
        if os.path.exists(upload_path):
            os.remove(upload_path)  # Replace existing file if exists

        fs.save("video.mp4", video_file)
        return HttpResponse("Video uploaded successfully!")

    return render(request, 'deepfake/upload_video.html')


# Upload photo
def process_photo(request):
    if request.method == "POST" and request.FILES.get("photo"):
        photo_file = request.FILES["photo"]

        # Define path to save the file
        upload_path = os.path.join('media', 'photos', 'photo.jpg')

        # Save the uploaded photo with a fixed filename
        fs = FileSystemStorage(location=os.path.dirname(upload_path))
        if os.path.exists(upload_path):
            os.remove(upload_path)  # Replace if file exists

        fs.save("photo.jpg", photo_file)
        return HttpResponse("Photo uploaded successfully!")

    return render(request, "upload_photo.html")


# Load the CNN model (TensorFlow)
model = tf.keras.models.load_model('ml_models/xception_deepfake_image.h5')


def process_input(request):
    if request.method == "POST" and request.FILES.get("input_file"):
        input_file = request.FILES["input_file"]

        # Save the uploaded file to the server
        fs = FileSystemStorage()
        file_path = fs.save(input_file.name, input_file)

        # Preprocess the input for your model (assuming image input)
        input_data = preprocess_input(file_path)

        # Make the prediction using the CNN model
        prediction = model.predict(input_data)

        # Return the prediction result
        return HttpResponse(f"Prediction Result: {prediction}")

    return render(request, "deepfake/upload.html")


def preprocess_input(file_path):
    # Assuming input is an image, use Keras to load and preprocess

    img = image.load_img(file_path, target_size=(224, 224))  # Adjust size based on your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

def detect_deepfake(request):
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']

        # Save the uploaded image temporarily
        temp_path = os.path.join(settings.MEDIA_ROOT, 'uploads', img.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb+') as destination:
            for chunk in img.chunks():
                destination.write(chunk)

        # Preprocess the image
        img_loaded = image.load_img(temp_path, target_size=(224, 224))  # Adjust size as needed
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize if your model requires it

        # Predict using the model
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            print(prediction[0][0])
            result = "The image is a Deepfake."
        else:
            print(prediction[0][0])
            result = "The image is Real."

        # Delete the temporary image file
        os.remove(temp_path)

    return render(request, 'deepfake/upload_photo.html', {'result': result})



rnn_model = tf.keras.models.load_model('ml_models/rnn_lstm.h5')
print(rnn_model.input_shape)


def detect_deepfake_video(request):
    result = None
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Save the uploaded video temporarily
        temp_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Extract frames from the video
        cap = cv2.VideoCapture(temp_path)
        frame_sequence = []
        sequence_length = 50  # Updated to match model input shape

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 128x128 to match model input size
            resized_frame = cv2.resize(frame, (128, 128))
            img_array = image.img_to_array(resized_frame)
            img_array /= 255.0  # Normalize the image if required by the model

            frame_sequence.append(img_array)

            # Stop if we reach the required sequence length
            if len(frame_sequence) == sequence_length:
                frame_batch = np.expand_dims(frame_sequence, axis=0)  # Shape: (1, 50, 128, 128, 3)

                # Predict using the RNN model
                prediction = rnn_model.predict(frame_batch)

                result = "The video is a Deepfake." if prediction[0][0] > 0.5 else "The video is Real."

                break  # Only process the first sequence for simplicity

        cap.release()
        os.remove(temp_path)

        if result is None:
            result = "Could not analyze the video. Please try another one."

    return render(request, 'deepfake/upload_video.html', {'result': result})

