import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN



# draw an image with detected objects
def draw_image_with_boxes_and_features(image, result_list):
    # load the image
    #data = plt.imread(filename)
    # plot the image
    plt.imshow(image)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    plt.axis('off')
    plt.show()
    
# Create the detector, using default weights
detector = MTCNN()

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector.detect_faces(rgb_frame)

    # Display faces on the original image
    draw_image_with_boxes_and_features(frame, faces)

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()    