# Drone Detection and Tracking

This project is designed to enable drone detection and movement tracking using PyTorch YOLOv8 and a Kalman Filter, respectively. Note that this project is not intended for detecting objects from a drone but solely for detecting and tracking drones from a video feed.

## Installation

To ensure compatibility and maintain package versions, it is highly recommended to use Python's virtual environment. Follow these steps to create and activate the virtual environment:

1. Create and activate the virtual environment:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

You will see `env>` at the start of your terminal or command prompt, indicating the virtual environment is active.

## Dataset

The project uses a custom [drone detection dataset](https://universe.roboflow.com/search?q=drone%20detection) from Roboflow, augmented with rotation and blur. The final dataset is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/jhilmitasri/dronedatasetaugmented/settings).

## Training

A [YOLOv8](https://docs.ultralytics.com/) model is employed for drone detection. You can refer to this Roboflow [blog post](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/) for training the model on a custom dataset.

Alternatively, the `drone-detection.ipynb` script was executed on Kaggle, utilizing 2 x GPU T4's capabilities. Although initially planned for 25 epochs, satisfactory results were achieved in just 20 epochs. The results are included in the repository.

## Tracking

The Kalman Filter method is utilized to track drone movements across the video feed. The [FilterPy](https://filterpy.readthedocs.io/en/latest/) library, which implements various Bayesian filters including Kalman filters, is used for this purpose.

## Implementation

To perform detection and tracking, run:

```bash
python tracked_detection.py
```

For detection only, run

```bash
python detect.py
```

**Note** - The `tracked_detection.py` script is designed to process multiple video files in the input_videos/ directory, making it suitable for general use in CCTV surveillance applications.


## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)