import cv2
import urllib.request

url, filename = ("https://images.unsplash.com/photo-1484353371297-d8cfd2895020?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NTUwfHxwZW9wbGV8ZW58MHx8MHx8fDA%3D", "scene.jpg")
urllib.request.urlretrieve(url, filename)    # Download the image

# Load the input image using OpenCV
image = cv2.imread("scene.jpg")


