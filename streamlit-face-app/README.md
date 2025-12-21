# Streamlit Face Recognition App

This project is a Streamlit application that allows users to upload a target image and a group image. The application uses a face recognition model to identify the target face in the group image and displays the resulting image with bounding boxes around detected faces.

## Project Structure

```
streamlit-face-app
├── src
│   ├── streamlit_app.py      # Main Streamlit application code
│   ├── face_utils.py         # Utility functions for face detection and embedding extraction
│   └── main.py               # Entry point for running the Streamlit app
├── requirements.txt           # List of dependencies
├── .gitignore                 # Files and directories to ignore by Git
└── README.md                  # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd streamlit-face-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guidelines

1. **Run the Streamlit app:**
   ```bash
   streamlit run src/main.py
   ```

2. **Upload Images:**
   - Upload a target image (preferably a single person).
   - Upload a group image containing multiple faces.

3. **View Results:**
   - The application will process the images and display the group image with bounding boxes around detected faces, indicating matches with the target image.

## Dependencies

- Streamlit
- DeepFace
- OpenCV
- MTCNN
- NumPy

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.