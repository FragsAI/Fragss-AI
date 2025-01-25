# Fragss-AI
# Video Segmentation Tool

The Video Segmentation Tool is a Python-based application that allows users to segment long video files into smaller clips based on various criteria, such as shot boundaries, object detection, and audio event detection.

## Features

- **Shot Boundary Detection**: Identify transitions between shots or scenes in the input video using histogram-based comparison techniques.
- **Video Clip Generation**: Extract video segments based on the detected shot boundaries and save them as individual clips.
- **Audio Clipping (Optional)**: Clip the audio of the input video based on the detected shot boundaries and save the audio segments separately.
- **Video and Audio Combination**: Combine the generated video and audio clips into the final output files.
- **GUI-based Interface**: Provide a user-friendly graphical interface for selecting the input video, output directory, and configuring the segmentation options.
- **Filtering Options**: Allow users to enable or disable object detection and audio event detection as additional filtering criteria for the video segmentation process.

## Prerequisites

- Python 3.6 or newer
- The following Python libraries:
  - `opencv-python`
  - `tkinter`
  - `moviepy`
  - `numpy`

## Usage
1. Instal the required libraries and run the program.
2. In the GUI, click the "Select Video" button to choose the input video file.
3. Click the "Select Output" button to choose the output directory where the segmented clips will be saved.
4. (Optional) Check the "Object Detection" and/or "Audio Event Detection" checkboxes to enable those filtering criteria.
5. Click the "Segment Video" button to start the video segmentation process.
6. The segmented video and audio clips will be saved in the selected output directory.

## Contributing

Contributions to the Video Segmentation Tool are welcome! If you find any issues or have ideas for improvements, please feel free to submit a pull request or open an issue on the [GitHub repository]

## Acknowledgments

- The `shot_boundary_detection`, `clip_video`, `clip_audio`, and `combine_video_audio` functions were adapted from the `Video_Segmentation_Tool.py` file.
- The GUI implementation was based on the Tkinter library.

## Thing to remember before running app.py:
1. Download FFmpeg:
- Download the Files: https://drive.google.com/drive/folders/1Ku9nnmQfBpeNI9M1HyvEoFQlbZorIX2Y?usp=drive_link
- Once downloaded, extract it to a folder, for example, C:\ffmpeg-master-latest-win64-gpl-shared\bin.
- Add FFmpeg to the System PATH:
- Right-click on "This PC" or "Computer" and select "Properties".
- Click on "Advanced system settings" and then click on the "Environment Variables" button.
- Under "System variables", find and select the Path variable, then click "Edit".
- In the "Edit Environment Variable" window, click "New" and add the path to the bin folder where you extracted FFmpeg (e.g., C:\ffmpeg-master-latest-win64-gpl-shared\bin).
- Click "OK" to close all windows.

2. ImageMagick
- Download and Install ImageMagick
- Run the installer and follow the installation steps.
- During the installation, make sure to:
- Select the option to add ImageMagick to your system's PATH ex: C:\Program Files\ImageMagick-7.1.1-Q16.
- Choose the "Install legacy utilities (e.g., convert)" option if you need older commands like convert.

## Function to check for existing fonts in the system ("to be executed in seperate file"):
    from matplotlib import font_manager

    # Get all available font paths
    font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    # Print font names and their paths
    print("Available Fonts:")
    for font_path in font_paths:
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        print(f"{font_name} : {font_path}")

