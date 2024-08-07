# Real-time Object Detection on League of Legends VODs with Custom YOLO Model (README NOT DONE YET)

This project captures frames from a specific window, processes them using a custom YOLO model for object detection, and displays the annotated frames in real-time.

## Requirements

- Python 3.6+
- OpenCV
- MSS
- NumPy
- PyGetWindow
- Threading
- Concurrent Futures
- Random
- Ultralytics YOLO
- Torch

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/real-time-object-detection.git
    cd real-time-object-detection
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt.txt
    ```

## Usage

1. **Update the model path:**

    Edit the script to include the path to your custom YOLO model:
    ```python
    model = YOLO("path/to/custom/model.pt").to(device)
    ```

2. **Run the script:**

    ```bash
    python real_time_object_detection.py
    ```

3. **Ensure the target window is open:**

    The script captures frames from a specific window title. Update the window title in the `get_window_bbox` function if needed:
    ```python
    def get_window_bbox(window_title="Your Window Title"):
    ```

## Notes

- The script uses threading to capture, process, and display frames concurrently.
- To stop the script, press `q` in the display window.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
