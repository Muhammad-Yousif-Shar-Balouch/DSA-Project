# Hand Gesture Control using Computer Vision  

This project demonstrates an innovative way to interact with your system using real-time hand gesture recognition powered by computer vision. With this program, you can control screen brightness, adjust system volume, zoom in/out, and even open random websites—all with simple hand gestures!  

## Features  
- **Brightness Control**: Adjust screen brightness dynamically using finger gestures.  
- **Volume Control**: Change the system's audio volume with hand movements.  
- **Zoom Functionality**: Perform zoom-in and zoom-out effects using gestures.  
- **Website Navigation**: Open random websites with a specific gesture.  
- **Gesture Feedback**: Visual feedback for detected gestures displayed on the screen.  

## Libraries Used  
- **OpenCV**: For video capture and image processing.  
- **MediaPipe**: For real-time hand detection and tracking.  
- **Screen Brightness Control**: To adjust screen brightness programmatically.  
- **PyCaw**: For controlling system audio.  
- **NumPy**: For mathematical computations.  
- **Scipy**: For advanced distance calculations.  

## How It Works  
1. The program captures live video using a webcam.  
2. MediaPipe detects hand landmarks in real-time.  
3. Specific gestures are mapped to system functions:  
   - Distance between thumb and index finger adjusts brightness.  
   - Distance between thumb and middle finger adjusts volume.  
   - Distance between thumb and ring finger controls zoom.  
   - Touching thumb to pinky opens a random website.  
4. Visual feedback for gestures is displayed on the screen, along with current brightness, volume, and zoom levels.  

## Setup and Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/hand-gesture-control.git
   ```
2. Install the required libraries:  
   ```bash
   pip install opencv-python mediapipe screen-brightness-control numpy pycaw scipy
   ```
3. Run the program:  
   ```bash
   python hand_gesture_control.py
   ```
4. Use the gestures described above to control the system features. Press **'q'** to exit the program.  

## Future Enhancements  
- Add support for custom gestures.  
- Expand functionality for controlling other system features.  
- Improve accuracy and gesture recognition.  

## Contributing  
Feel free to fork this repository and contribute improvements!  

## License  
This project is licensed under the MIT License.  
