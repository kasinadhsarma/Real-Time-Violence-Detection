# Real-Time-Violence-Detection

**Understanding the Code's Purpose:**

- The main goal of this code is to build a system that can detect violence in videos.

**It does this in several steps:**
1. Prepares video data for the model.
2. Builds a deep learning model to learn patterns of violence.
3. Trains the model to recognize these patterns.
4. It uses the trained model to predict whether a new video contains violence.

**Simplified Explanation of Key Parts:**

1. **Libraries:**
   - OpenCV (cv2) is used to work with videos and images.
   - NumPy (numpy) is used for handling numbers and arrays, which videos are represented as.
   - Keras (keras) is used to build and train the deep learning model.

2. **Preparing the Videos:**
   - The code finds video files representing violence and non-violence.
   - It breaks each video into individual frames.
   - Each frame is resized to a standard size to make it easier for the model to process.

3. **The Model:**
   - Model Type: The model uses a combination of image-based techniques (MobileNetV2) and sequence-based techniques (LSTM) to learn from video sequences.
   - Training: The model is shown many video examples and gradually learns to identify patterns that indicate violence.

4. **Making Predictions:**
   - New Video: The code breaks it into frames when given a new video.
   - Prediction: The model analyzes the sequence of frames and decides if the video likely contains violence or not.

**Let's Simplify Even More:**

**5. Think of it like this:**
   - The code teaches a computer how to "watch" videos.
   - It shows the computer many examples of violent and non-violent videos.
   - The computer learns to spot the differences between violence and non-violence.
   - Now, the computer can watch new videos and tell you if they seem violent.

**How to test the code:**

```
git clone https://github.com/kasinadhsarma/Real-Time-Violence-Detection
cd Real-Time-Violence-Detection/
code .
```

The tools and technologies utilized for the project are idx.google.com, Ubuntu Linux LTS 22.04, Kaggle for datasets and sample code collection, integrity making, and research. The developers involved in experiments and implementations for fine-tuning with API keys in the future are myself and bmentech343, which is my Gmail account.

**making experiments implementations in fine-tuning with API key in future**
  - openai
  - gemini
  - claude3
