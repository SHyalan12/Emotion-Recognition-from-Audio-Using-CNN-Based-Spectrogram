##Emotion Recognition from Audio Using CNN-Based Spectrogram Analysis


This project consists of Python notebooks designed to run on Google Colab, utilizing datasets stored on Google Drive. Follow the instructions below for proper setup and execution.

## Folder Structure

### Dataset
- Contains raw audio data for song and speech, used for model training and validation.

### Data Processing and Model
- Notebooks for converting audio clips into spectrograms (separately for song, speech, and combined datasets).
- Includes basic CNN model experiments.
- Contains generated spectrograms.

### Testing Data Processing
- Notebooks for creating augmented datasets:
  - Noise
  - Pitch
  - Speed

### Data Augmentation Testing
- Files for evaluating augmented datasets using the basic CNN model.

### Augmented Dataset
- Due to size constraints, access the [Augmented Dataset here](https://drive.google.com/drive/folders/1EHp8myJrz8bJFW0EGLLXC_uJVnd_IxmJ).
- Contains intermediate augmented spectrograms for song and speech.

### Final Augmented Dataset
- Notebooks demonstrating selection and combination methods for the final augmented datasets.
- Final augmented spectrograms for song and speech.

### Final Model
- Notebooks for training and evaluating final song and speech models.
- Includes notebook specifically for real-world audio testing.

## Project Scope Adjustment
Initial combined modeling of song and speech resulted in low accuracy. Therefore, the current focus is on separate song and speech models, conducting comparative experiments instead of the initially proposed short vs. long audio clips.

## Song Model Workflow
- **Testing Models:** Experiments on dropout, early stopping, learning rates, layers architecture, optimizer methods.
- **Selecting Models:** Best-performing model identification.
- **Data Augmentation Testing:** Determining effective augmentation levels.
- **Final Dataset:** Generate final augmented dataset for the selected model.
- **Fine Tuning:** Transfer learning attempted but ineffective.
- **Final Evaluation:** Train and evaluate the selected final model.

## Speech Model Workflow
Similar to Song workflow, with additional real-world audio clip testing step:
- **Testing Models:** Dropout, early stopping, learning rates, layers, optimizer methods.
- **Selecting Models:** Identify optimal model.
- **Data Augmentation Testing:** Optimal augmentation level identification.
- **Final Dataset:** Produce final augmented dataset.
- **Fine Tuning:** Transfer learning attempted but ineffective.
- **Final Evaluation:** Model training and evaluation.
- **Real-world Audio Testing:** Test final model on real-world audio samples.

## Execution Instructions

1. Upload all files to Google Drive. *(Only `Final_Model` and `Final_Augmented_Dataset` are needed for running the final model)*  
   - `augmented_spectrograms_dataset_song` is the dataset for `Final_model_song`  
   - `augmented_spectrograms_dataset_speech` is the dataset for `Final_model_speech`  
   - `Lecture_recording.png` can be used for testing real-world recording

2. Verify and adjust the data path in each Python notebook as needed.

3. Run the notebooks directly in Google Colab.
