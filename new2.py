import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import webbrowser
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from scipy.spatial.distance import euclidean

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils



# Pycaw setup for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control = cast(interface, POINTER(IAudioEndpointVolume))

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize control variables
brightness = 50
volume = 50
zoom = 100
website_opened = False  # To prevent repeated website opening

# Random website list
websites = ["https://www.google.com", "https://www.github.com", "https://www.wikipedia.org"]

# Function to open a random website
def open_random_website():
    global website_opened
    if not website_opened:
        webbrowser.open(np.random.choice(websites))
        website_opened = True  # Set to true to avoid repeated opening

# Function to reset website_opened flag
def reset_website_opened():
    global website_opened
    website_opened = False

# Core loop for frame capture and gesture recognition
while True:
    _, frame = cap.read()

    # Flip the image horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB format
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []

    # If hands are detected
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            # Draw the landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList:
        # Thumb coordinates
        x_thumb, y_thumb = landmarkList[4][1], landmarkList[4][2]

        # Index finger coordinates (Brightness)
        x_index, y_index = landmarkList[8][1], landmarkList[8][2]
        distance_index_thumb = int(euclidean((x_index, y_index), (x_thumb, y_thumb)))
        new_brightness = np.interp(distance_index_thumb, [15, 220], [0, 100])
        if abs(new_brightness - brightness) > 1:
            brightness = int(new_brightness)
            sbc.set_brightness(brightness)

        # Middle finger coordinates (Volume)
        x_middle, y_middle = landmarkList[12][1], landmarkList[12][2]
        distance_middle_thumb = int(euclidean((x_middle, y_middle), (x_thumb, y_thumb)))
        volume_level = np.interp(distance_middle_thumb, [15, 220], [0, 100])  # Volume range
        if abs(volume - volume_level) > 1:
            volume = int(volume_level)
            volume_control.SetMasterVolumeLevel(np.interp(volume, [0, 100], [-65, 0]), None)

        # Ring finger coordinates (Zoom)
        x_ring, y_ring = landmarkList[16][1], landmarkList[16][2]
        distance_ring_thumb = int(euclidean((x_ring, y_ring), (x_thumb, y_thumb)))
        new_zoom = np.interp(distance_ring_thumb, [15, 220], [50, 200])
        zoom = int(new_zoom)

        # Pinky finger coordinates (Website open)
        x_pinky, y_pinky = landmarkList[20][1], landmarkList[20][2]
        distance_pinky_thumb = int(euclidean((x_pinky, y_pinky), (x_thumb, y_thumb)))
        if distance_pinky_thumb < 30:  # Adjust threshold as needed
            open_random_website()
        else:
            reset_website_opened()

        # Draw lines and display distances
        cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0), 2)  # Green for Brightness
        cv2.putText(frame, f'{distance_index_thumb}px', ((x_thumb + x_index) // 2, (y_thumb + y_index) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.line(frame, (x_thumb, y_thumb), (x_middle, y_middle), (0, 255, 255), 2)  # Yellow for Volume
        cv2.putText(frame, f'{distance_middle_thumb}px', ((x_thumb + x_middle) // 2, (y_thumb + y_middle) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.line(frame, (x_thumb, y_thumb), (x_ring, y_ring), (255, 0, 0), 2)  # Blue for Zoom
        cv2.putText(frame, f'{distance_ring_thumb}px', ((x_thumb + x_ring) // 2, (y_thumb + y_ring) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.line(frame, (x_thumb, y_thumb), (x_pinky, y_pinky), (255, 255, 255), 2)  # White for Website
        cv2.putText(frame, f'{distance_pinky_thumb}px', ((x_thumb + x_pinky) // 2, (y_thumb + y_pinky) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display values for brightness, volume, and zoom
        cv2.putText(frame, f'Brightness: {brightness}%', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Volume: {volume}%', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f'Zoom: {zoom}%', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Apply zoom effect by cropping and resizing
    zoom_factor = zoom / 100
    centerX, centerY = frame.shape[1] // 2, frame.shape[0] // 2
    radiusX, radiusY = int(centerX / zoom_factor), int(centerY / zoom_factor)

    minX, maxX = max(0, centerX - radiusX), min(frame.shape[1], centerX + radiusX)
    minY, maxY = max(0, centerY - radiusY), min(frame.shape[0], centerY + radiusY)

    cropped_frame = frame[minY:maxY, minX:maxX]
    frame = cv2.resize(cropped_frame, (frame.shape[1], frame.shape[0]))

    # Show the current frame
    cv2.imshow("Hand Gesture Control", frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF

    # Exit the program (press 'q')
    if key == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
