# License Plate Recognition Project

This project implements a vehicle detection and license plate recognition system using YOLOv8 and EasyOCR. It processes video files to detect vehicles, extract license plates, and apply optical character recognition (OCR) to read the plate numbers. The detected license plates are then overlaid on the video frames with bounding boxes around both the vehicles and their corresponding plates.

## Features

- Vehicle detection using YOLOv8
- License plate extraction and recognition with EasyOCR
- Visualization of detected license plates with bounding boxes
- Interpolation of missing data to smooth output

## Links

- **Test Videos:** [Google Drive](https://drive.google.com/drive/folders/1-evm7MTQeDoDXC7kdhunhYagXbO5QXqJ?usp=sharing)
- **Dataset:** [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
- **Sample Output:** [Google Drive](https://drive.google.com/file/d/15iArRGwCIQOhGGQxkYvbKcAAFrGeMyz8/view?usp=sharing)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mohamedd-adel/LicensePlateDetector.git
   cd LicensePlateDetector
   ```

2. Install the SORT module:
   Download from [SORT GitHub](https://github.com/abewley/sort).

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run `main.py` with a sample video file to generate the `test.csv` file:
   ```bash
   python main.py
   ```

2. Run `add_missing_data.py` for interpolation of values to match up for the missing frames and smooth output:
   ```bash
   python add_missing_data.py
   ```

3. Finally, run `visualize.py`, passing in the interpolated CSV files to obtain a smooth output for license plate detection:
   ```bash
   python visualize.py
   ```
   ### Important Note

Make sure to change the video path in both `main.py` and `visualize.py` to the path where you downloaded the video files. 



## Future Enhancements

To improve the accuracy of the extracted text from EasyOCR, future enhancements will focus on refining the OCR model and exploring additional preprocessing techniques to enhance text recognition performance, particularly in challenging conditions such as varying lighting and angles.
