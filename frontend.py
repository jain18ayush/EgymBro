import streamlit as st
import tempfile
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import model
import critiqueModel as critique

st.header("E-GymBro")
st.markdown("A site to track your fitness gains quantitatively...")

def classify_angles(new_angles, model_filename):
    # Load the trained model from the file
    knn = joblib.load(model_filename)
    
    # Calculate features for the new set of angles
    new_features = calculate_features(new_angles)
    
    # Classify the new set of angles
    prediction = knn.predict([new_features])
    
    return prediction[0]


def upload_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    video_file_path = tfile.name  # Get the file path of the temporary file
    angles = model.processVideo(video_path=video_file_path)
    result = critique.classify_angles(angles, 'knn_model.joblib')
    st.write(result)
    # # Load the video file
    # video = cv2.VideoCapture(video_file_path)

    # # Check if video file opened successfully
    # if not video.isOpened():
    #     st.error("Could not open the video file.")
    #     return

    # # Get video frame count and fps
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(video.get(cv2.CAP_PROP_FPS))

    # # Process and display each frame
    # for i in range(frame_count):
    #     ret, frame = video.read()
    #     if ret:
    #         # Convert the frame to RGB (OpenCV uses BGR by default)
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         st.image(frame_rgb, channels="RGB")
    #         # Sleep for the appropriate amount of time to achieve the original video speed
    #         time.sleep(1.0 / fps)  # Corrected line
    #     else:
    #         st.warning("Failed to retrieve frame.")
    #         break

    # Release the video file

# Upload the video file or capture a live recording
# option = st.radio("Choose an option:", ["Upload a video file", "Capture a live recording"])
# if option == "Upload a video file":
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    upload_video(uploaded_file)


st.header("About")
st.markdown("<h4>The Science Behind Muscle Growth</h4>", unsafe_allow_html=True)
st.markdown("The fundmentals of muscle growth is reliant on the repetitive tearing and repairing of microscopic muscle fibers." + 
            " When lifting heavy weights, your muscle fibers are strained and torn, creating the pain and weakness you feel during a bicep curl, for instance." +
            " The soreness after is a result of your body repairing these tears, making stronger bonds and thus increasing the size of your muscles. " +
            " This process is known as hypertrophy, and going to the gym can help target very specific groups of muscles in body.")
st.markdown("<h4>The Proper Form for Benching</h4>", unsafe_allow_html=True)
st.markdown("Having proper form is essential to having a proper workout. Proper form requries smooth and correct motions to target the desired muscle group." +
            " For instance, to perform a perfect bench press, you must:\n" + 
            "- Avoid flaring out your elbows to reduce the strain on your shoulders\n" +
            "- Arch your back to establish tension in your back for more power\n" + 
            "- Keep your feet flat on the ground to establish a leg drive\n" + 
            "- The bar must be brought down below the chest in an arc-like motion\n" + 
            "\nFulfulling all these requirements is necessary to properly target the chest (specifically the pectoralis major), but it may be hard for a beginner, " +
            "especially one with no professional guidance, to figure this out.\n\nHence...")
st.markdown("<h4>What We Offer</h4>", unsafe_allow_html=True)
st.markdown("Using pose estimation with our Python algorithm, we offer you a solution to track the accuracy of your form with footage of professional bodybuilder. " +
            "By comparing your video footage and our own data, we can compare the key quantitative distinctions between your technique and that of a pro. Try the model above to see " +
            "how accurate your technique is and potentially use our prescribed feedback to improve your form! Check out our links below to learn more about bodybuilding, weightlifting, and computer vision!")
st.header("Links")
st.markdown("Pose estimation using Google Mediapipe --> https://developers.google.com/mediapipe")
st.markdown("The science behind bodybuilding --> https://www.builtlean.com/muscles-grow/")
st.markdown("Maintaining form for weighted movements --> https://www.bodybuilding.com/fun/likness25.htm")