import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image_path):
    
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        return
    

    img = cv2.imread(image_path)
    
    
    if img is None:
        print("Error: Unable to load the image. Please check the file path.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_cont=len(faces)
    print(f"Number of face detected is:{face_cont}")
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
   
    cv2.imshow('Faces Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()


if __name__ == '__main__':
    
    image_path = 'C:/Users/tapas/OneDrive/Desktop/face/ag.jpg'
    detect_faces(image_path)
