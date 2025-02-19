# Combined Landslide Classification System

The Combined Landslide Classification System provides three different methodologies to classify landslide events:  
1. **Image-Based Classification (CNN)**  
2. **Geometric-Based Classification (XGBoost)**  
3. **TDA-Based Classification (XGBoost with Topological Data Analysis)**

This repository includes a complete pipeline—from data preprocessing and feature extraction to machine learning-based prediction—for classifying landslides triggered by earthquakes or rainfall.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running the Script](#running-the-script)
  - [Classification Methods](#classification-methods)
- [Screenshots & Demo Video](#screenshots--demo-video)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Multiple Classification Approaches:**  
  - **CNN-based image classification:** Convert landslide polygons to grayscale images and classify using a pre-trained CNN.
  - **Geometric-based classification:** Extract geometric properties (e.g., area, perimeter) and classify with XGBoost.
  - **TDA-based classification:** Generate 3D point clouds from DEM data, extract topological features, and classify with XGBoost.
  
- **DEM Data Integration:**  
  Automatically downloads and processes Digital Elevation Model (DEM) data required for TDA-based classification.

- **User-friendly Interface:**  
  Interactive command-line prompts guide you through selecting the region and classification method.

- **Model Training & Prediction:**  
  Train new models or use pre-saved ones for rapid prediction.

---

## Requirements

- **Python 3.7 or higher**

- **Key Python Packages:**
  - `numpy`
  - `geopandas`
  - `joblib`
  - `matplotlib`
  - `osgeo` (GDAL)
  - `elevation`
  - `scikit-learn`
  - `xgboost`

- **Custom Modules:**
  - `Image_model`
  - `Geometric_Model`
  - `topological_features_based_modell`

> **Note:** Ensure that the above custom modules are included in your repository or accessible via your Python path.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/landslide-classification.git
   cd landslide-classification
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Custom Modules:**

   Ensure that the custom modules (`Image_model.py`, `Geometric_Model.py`, and `topological_features_based_modell.py`) are in the same directory as the main script or correctly referenced in your `PYTHONPATH`.

---

## Usage

### Data Preparation

- **Shapefiles:**  
  Prepare the shapefile data for your region of interest. Example paths are provided in the script (e.g., for Greece, US landslides, and Japan training datasets). Adjust these paths to point to your local data files.

- **DEM Data:**  
  The script automatically downloads DEM tiles using the [elevation](https://github.com/Anaconda-Platform/elevation) package. Ensure that your device has sufficient RAM if processing a large region.

### Running the Script

To run the combined classification system, execute the main Python script:

```bash
python main.py
```

Follow the on-screen prompts to select:
- The **classification method** (Image-based, Geometric-based, or TDA-based).
- The **region** for testing (e.g., Greece, US, or Puerto Rico).
- Additional options such as training a new model (for TDA-based classification) or using a pre-saved model.

### Classification Methods

#### 1. Image-Based Classification (CNN)

- **Workflow:**  
  - Select region (Greece or US).
  - Load and preprocess shapefile polygons.
  - Convert polygons to images.
  - Classify using a pre-trained CNN.
  - Visualize sample images and prediction results.

#### 2. Geometric-Based Classification (XGBoost)

- **Workflow:**  
  - Select region.
  - Load shapefiles and extract geometric features.
  - Train or use a pre-trained XGBoost classifier.
  - Display classification results based on geometric properties.

#### 3. TDA-Based Classification (XGBoost)

- **Workflow:**  
  - Choose to **train a new model** or use an existing one.
  - Download DEM data if not already present.
  - Create 3D point clouds from polygon geometries.
  - Extract topological features using TDA.
  - Classify using an XGBoost model.
  - View probabilities for landslides triggered by earthquake vs. rainfall.

---

## Screenshots & Demo Video

### Screenshots

#### Main Menu Prompt
![Main Menu](screenshots/main_menu.png)

#### Image-Based Classification Results
![Image Classification](screenshots/image_classification.png)

#### Geometric-Based Classification Results
![Geometric Classification](screenshots/geometric_classification.png)

#### TDA-Based Classification Workflow
![TDA Classification](screenshots/s1.png)

### Demo Video

Watch the demo video on how to run and use the system:

[![Watch Demo](screenshots/video_thumbnail.png)](https://www.youtube.com/watch?v=your_demo_video_link)

---

## Contributing

Contributions are welcome! If you want to enhance the system, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeatureName`.
5. Submit a pull request.

For any issues or suggestions, please open an issue in the repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please contact:

- **Snehal** – [snehal@flaxandteal.co.uk](mailto:snehal@flaxandteal.co.uk)
- **Project Link:** [https://github.com/yourusername/landslide-classification](https://github.com/yourusername/landslide-classification)
