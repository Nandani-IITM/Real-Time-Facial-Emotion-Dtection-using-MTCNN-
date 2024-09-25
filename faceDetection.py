import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_image_with_boxes(image, result_list):
    # Plot the image
    plt.imshow(image)

    # Get the context for drawing boxes
    ax = plt.gca()

    # Plot each box
    for result in result_list:
        # Get coordinates
        x, y, width, height = result['box']

        # Create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')

        # Draw the box
        ax.add_patch(rect)

    # Show the plot
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
    draw_image_with_boxes(frame, faces)

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
