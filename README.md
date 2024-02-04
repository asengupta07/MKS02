# Student Engagement Tracker

## Approach
- **Face Detection using dlib:**
  - Employ the 'Human Face Detector' model from the dlib library to identify and locate faces within the video feed.
  - Utilize the face coordinates obtained to crop the facial region for further analysis.

- **Data Preprocessing:**
  - Normalize and preprocess the cropped facial images to ensure consistency and enhance the model's interpretability.
  - Apply any necessary data augmentation techniques to improve model generalization.

- **Fine-tuned CNN Model:**
  - Develop a Convolutional Neural Network (CNN) based on the ResNet50 architecture.
  - Fine-tune the model on a relevant dummy dataset, incorporating a diverse range of facial expressions to enhance its ability to recognize various engagement states.

- **Engagement Prediction:**
  - Train the model to classify the extracted features to detect whether the student is engaged or not.
  - Validate the model's performance using a dummy evaluation dataset to ensure its accuracy and generalization capabilities.

- **Real-time Video Analysis:**
  - Implement a real-time video analysis module that continuously captures frames from the video feed.
  - Apply the trained model to the cropped facial regions to predict student engagement at regular intervals during online classes.
- **Engagement Report Generation:**
  - Accumulate the predictions over time to generate statistics on student engagement.
  - Create visualizations, such as graphs or charts, to present the data in an easily interpretable format.
  - Include metrics such as average engagement, engagement fluctuations, and duration of sustained attention.

- **Application Development:**
  - Develop a user-friendly application that integrates the entire student engagement detection system.
  - Implement features for configuring analysis intervals, visualizing real-time engagement data, and generating comprehensive reports.

- **Deployment and Integration:**
  - Deploy the application to be seamlessly integrated into online class platforms.
  - Ensure compatibility with various video conferencing tools and adaptability to different hardware configurations.

- **Continuous Improvement:**
  - Regularly update the model with new data to adapt to evolving student engagement patterns.
  - Gather feedback from educators and users to identify areas for improvement and implement necessary enhancements in subsequent versions of the system.

## Deployed Demo:
[HuggingFace Spaces Demo Link](https://huggingface.co/spaces/asengupta07/EngagementTracker)

## Contributors:
- Arnab Sengupta
- Srijan Sarkar
- Sachindra Kumar Singh
- Ankana Datta
