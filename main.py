

import os
import time
import random
import numpy as np
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from osgeo import gdal
import elevation

# -----------------------------
# IMPORTS FOR IMAGE-BASED CLASSIFICATION
# -----------------------------
from Image_model import (
    read_shapefiles as read_shapefiles_img,
    latlon_to_eastnorth,
    make_ls_images,
    increase_resolution_polygon,
    visualize_images,
    train_augment,
    classify_inventory_cnn,
    visualize_all as visualize_all_img
)

# -----------------------------
# IMPORTS FOR GEOMETRIC-BASED CLASSIFICATION
# -----------------------------
from Geometric_Model import (
    read_shapefiles as read_shapefiles_geo,
    latlon_to_eastnorth as latlon_to_eastnorth_geo,
    get_geometric_properties_landslide,
    classify_inventory_xgb,
    visualize_all as visualize_all_geo
)

# -----------------------------
# IMPORTS FOR TDA-BASED CLASSIFICATION
# -----------------------------
from topological_features_based_modell import (
    read_shapefiles as read_shapefiles_tda,
    plot_polygon,
    make_3d_polygons,
    get_ml_features_with_files,
    classify_inventory_tda_with_xgboost,
    min_max_inventory,
)

# -------------------------------------------------------------------
# TDA-BASED: Additional Helper Functions (download, train, predict)
# -------------------------------------------------------------------
def download_dem(poly_data, dem_location, inventory_name):
    """
    Download the DEM corresponding to the inventory region.
    """
    longitude_min, longitude_max, latitude_min, latitude_max = min_max_inventory(poly_data, 0.00, -0.00)
    total_number_of_tiles = (longitude_max - longitude_min) * (latitude_max - latitude_min)
    print('Total number of DEM tiles:', total_number_of_tiles)
    print("** Ensure that the number of tiles is manageable for your device RAM **")
    
    final_output_filename = os.path.join(dem_location, inventory_name)
    
    if total_number_of_tiles < 10:
        longitude_min, longitude_max = longitude_min - 0.4, longitude_max + 0.4
        latitude_min, latitude_max = latitude_min - 0.4, latitude_max + 0.4
        print("Downloading DEM for a small region (less than 10 tiles).")
        elevation.clip(bounds=(longitude_min, latitude_min, longitude_max, latitude_max), output=final_output_filename)
        elevation.clean() 
    else:
        print('Downloading DEM for a larger region (more than 10 tiles).')
        latitude_width = int(latitude_max - latitude_min)
        longitude_width = int(longitude_max - longitude_min)
        add_latitude = 3 - latitude_width % 3
        add_longitude = 3 - longitude_width % 3
        latitude_max += add_latitude
        longitude_max += add_longitude
        latitude_width = (latitude_max - latitude_min)
        longitude_width = (longitude_max - longitude_min)
        t = 0
        for j in range(0, latitude_width, 3):
            for i in range(0, longitude_width, 3):
                t += 1
                output = os.path.join(dem_location, f'inven_name{t}.tif')
                elevation.clip(bounds=(longitude_min + i, latitude_max - j - 3, longitude_min + i + 3, latitude_max - j), output=output)    
                elevation.clean()
        NN = 10800
        DEM_DATA = np.zeros((NN * latitude_width // 3, NN * longitude_width // 3), dtype='uint16')
        t = 1
        X_0, Y_0 = [], []
        for i in range(latitude_width // 3):
            for j in range(longitude_width // 3):
                inv_name = f"inven_name{t}.tif"
                data_name = os.path.join(dem_location, inv_name)
                DEM = gdal.Open(data_name)
                x_0, x_res, _, y_0, _, y_res = DEM.GetGeoTransform()
                X_0.append(x_0)
                Y_0.append(y_0)
                print(x_0, x_res, _, y_0, _, y_res)
                z = gdal.Dataset.ReadAsArray(DEM)
                DEM_DATA[(i * NN):(i * NN) + NN, (j * NN):(j * NN) + NN] = z
                t += 1
                print("Tile:", t)
        x_0 = min(X_0)
        y_0 = max(Y_0)
        time.sleep(180)  # Pause if needed for system resources
        geotransform = (x_0, x_res, 0, y_0, 0, y_res)
        driver = gdal.GetDriverByName('Gtiff')
        final_output_filename = os.path.join(dem_location, inventory_name)
        dataset = driver.Create(final_output_filename, DEM_DATA.shape[1], DEM_DATA.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotransform)
        dataset.GetRasterBand(1).WriteArray(DEM_DATA)
    time.sleep(180)
    return final_output_filename

def train_xgb_model(eq_features, rf_features, model_path, scaler_path, feat_idx_path):
    """
    Train an XGBoost model on earthquake (label 0) and rainfall (label 1) features.
    The model is trained with a feature selection step (top 10 features) and saved.
    """
    # Remove non-numeric first column if present
    if isinstance(eq_features[0, 0], str):
        eq_features = eq_features[:, 1:]
    if isinstance(rf_features[0, 0], str):
        rf_features = rf_features[:, 1:]
    eq_features = eq_features.astype(float)
    rf_features = rf_features.astype(float)
    
    eq_label = np.zeros((eq_features.shape[0], 1))
    rf_label = np.ones((rf_features.shape[0], 1))
    
    n1, n2 = eq_features.shape[0], rf_features.shape[0]
    # Balance dataset by undersampling the larger class
    if n1 > n2:
        inds = random.sample(range(n1), n2)
        train_eq = eq_features[inds, :]
        train_label_eq = eq_label[inds]
        train_rf = rf_features
        train_label_rf = rf_label
    else:
        inds = random.sample(range(n2), n1)
        train_rf = rf_features[inds, :]
        train_label_rf = rf_label[inds]
        train_eq = eq_features
        train_label_eq = eq_label
        
    train_data = np.vstack((train_eq, train_rf))
    train_label = np.vstack((train_label_eq, train_label_rf))
    
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    print("Train data scaled shape:", train_data_scaled.shape)
    
    best_params = {
        "max_depth": 5,
        "learning_rate": 0.2,
        "n_estimators": 780,
        "gamma": 20,
        "subsample": 0.57,
        "colsample_bytree": 0.9498,
        "alpha": 5,
        "lambda": 5,
        "min_child_weight": 5,
        "scale_pos_weight": 0.63626821
    }
    
    clf = XGBClassifier(**best_params, eval_metric="logloss")
    clf.fit(train_data_scaled, train_label.ravel())
    feature_importance = clf.feature_importances_
    indices = np.argsort(-feature_importance)[:10]
    print("Selected top feature indices:", indices)
    
    clf_final = XGBClassifier(**best_params, eval_metric="logloss")
    clf_final.fit(train_data_scaled[:, indices], train_label.ravel())
    
    joblib.dump(clf_final, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(indices, feat_idx_path)
    
    print("Training complete. Model saved to disk.")
    return clf_final, scaler, indices

def predict_xgb_model(test_features, model, scaler, indices):
    """
    Predict class labels and probabilities for test features using a saved model.
    """
    if isinstance(test_features[0, 0], str):
        test_features = test_features[:, 1:]
    test_features = test_features.astype(float)
    
    test_data_scaled = scaler.transform(test_features)
    print("Test data scaled shape:", test_data_scaled.shape)
    preds = model.predict(test_data_scaled[:, indices])
    pred_probs = model.predict_proba(test_data_scaled[:, indices])
    print("Predictions:", preds)
    return preds, pred_probs

# --------------------------------------------------
# Function: Image-Based Classification Workflow
# --------------------------------------------------
def run_image_based_classification():
    print("You selected Image-Based Classification")
    print("Select the region for testing landslides:")
    print("1. Greece")
    print("2. US (Puerto Rico - Hurricane Maria)")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("You selected Greece landslides.")
        shapefile_path = "/home/snehal/Downloads/landsifier_fresh/Final Model/Greece inventory/ground_failure_polygons.shp"
        ground_failures = read_shapefiles_img(shapefile_path)
        ground_failures = ground_failures[ground_failures['type'] != 'Liquefaction'].dropna(subset=['geometry'])
        test_inventory = ground_failures[ground_failures['epicentral'] == 'Greece'].reset_index(drop=True)
    elif choice == "2":
        print("You selected US (Puerto Rico - Hurricane Maria) landslides.")
        us_shapefile_path = "/home/snehal/Downloads/landsifier_fresh/Final Model/US_Landslide_v3_shp/us_ls_v3_poly.shp"
        us_data = read_shapefiles_img(us_shapefile_path)
        test_inventory = us_data[us_data['Inventory'] == 'USGS PR Hurricane Maria (Bessette-Kirton and others, 2019)'].reset_index(drop=True)
    else:
        print("Invalid selection. Exiting image-based classification.")
        return
    
    # Convert landslide polygons to images
    test_inventory_features = make_ls_images(test_inventory)
    visualize_images(test_inventory_features, num_images=5)
    
    # Load Japan training datasets (earthquake & rainfall) and extract images
    earth_hokkaido_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Earthquake_hokkaido_polygons.shp")
    earth_iwata_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Earthquake_iwata_polygons.shp")
    earth_niigata_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Earthquake_niigata_polygons.shp")
    rain_kumamoto_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Rainfall_kumamoto_polygons.shp")
    rain_fukuoka_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Rainfall_fukuoka_polygons.shp")
    rain_saka_shp = read_shapefiles_img("/home/snehal/Downloads/landsifier_fresh/Final Model/Japan Inventory/Rainfall_saka_polygons.shp")
    
    features_earth_hokkaido = make_ls_images(earth_hokkaido_shp)
    features_earth_iwata = make_ls_images(earth_iwata_shp)
    features_earth_niigata = make_ls_images(earth_niigata_shp)
    features_rain_kumamoto = make_ls_images(rain_kumamoto_shp)
    features_rain_fukuoka = make_ls_images(rain_fukuoka_shp)
    features_rain_saka = make_ls_images(rain_saka_shp)
    
    Total_earthquake_inventory_features_Japan = np.vstack((features_earth_hokkaido, features_earth_iwata, features_earth_niigata))
    Total_rainfall_inventory_features_Japan = np.vstack((features_rain_kumamoto, features_rain_fukuoka, features_rain_saka))
    
    # Classify test inventory with the CNN model
    predict_probability = classify_inventory_cnn(
        Total_earthquake_inventory_features_Japan,
        Total_rainfall_inventory_features_Japan,
        test_inventory_features
    )
    
    visualize_all_img(predict_probability)

# --------------------------------------------------
# Function: Geometric-Based Classification Workflow
# --------------------------------------------------
def run_geometric_based_classification():
    print("You selected Geometric-Based Classification")
    print("Select the region for testing landslides:")
    print("1. Greece")
    print("2. US (Puerto Rico - Hurricane Maria)")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("You selected Greece landslides.")
        shapefile_path = "/home/snehal/Downloads/landsifier_fresh/Final Model/Greece inventory/ground_failure_polygons.shp"
        ground_failures = read_shapefiles_geo(shapefile_path)
        ground_failures = ground_failures[ground_failures['type'] != 'Liquefaction'].dropna(subset=['geometry'])
        test_inventory = ground_failures[ground_failures['epicentral'] == 'Greece'].reset_index(drop=True)
    elif choice == "2":
        print("You selected US (Puerto Rico - Hurricane Maria) landslides.")
        us_shapefile_path = "/Users/deepaksingh/Documents/vscode/snehal/US_Landslide_v3_shp/us_ls_v3_poly.shp"
        us_data = read_shapefiles_geo(us_shapefile_path)
        test_inventory = us_data[us_data['Inventory'] == 'USGS PR Hurricane Maria (Bessette-Kirton and others, 2019)'].reset_index(drop=True)
    else:
        print("Invalid selection. Exiting geometric-based classification.")
        return
    
    # Extract geometric features from the test inventory
    test_inventory_features = get_geometric_properties_landslide(test_inventory)
    
    # Load Japan training datasets and extract geometric features
    earth_hokkaido_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_hokkaido_polygons.shp")
    earth_iwata_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_iwata_polygons.shp")
    earth_niigata_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_niigata_polygons.shp")
    rain_kumamoto_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_kumamoto_polygons.shp")
    rain_fukuoka_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_fukuoka_polygons.shp")
    rain_saka_shp = read_shapefiles_geo("/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_saka_polygons.shp")
    
    features_earth_hokkaido = get_geometric_properties_landslide(earth_hokkaido_shp)
    features_earth_iwata = get_geometric_properties_landslide(earth_iwata_shp)
    features_earth_niigata = get_geometric_properties_landslide(earth_niigata_shp)
    features_rain_kumamoto = get_geometric_properties_landslide(rain_kumamoto_shp)
    features_rain_fukuoka = get_geometric_properties_landslide(rain_fukuoka_shp)
    features_rain_saka = get_geometric_properties_landslide(rain_saka_shp)
    
    Total_earthquake_inventory_features_Japan = np.vstack((features_earth_hokkaido, features_earth_iwata, features_earth_niigata))
    Total_rainfall_inventory_features_Japan = np.vstack((features_rain_kumamoto, features_rain_fukuoka, features_rain_saka))
    
    # Classify using the geometric features with XGBoost
    predict_probability = classify_inventory_xgb(
        Total_earthquake_inventory_features_Japan,
        Total_rainfall_inventory_features_Japan,
        test_inventory_features
    )
    
    visualize_all_geo(predict_probability)

# --------------------------------------------------
# Function: TDA-Based Classification Workflow
# --------------------------------------------------
def run_tda_based_classification():
    print("You selected TDA-Based Classification")
    print("Select mode:")
    print("1. Train new model")
    print("2. Use pre-saved model for testing")
    mode = input("Enter 1 or 2: ").strip()
    
    model_path = "xgb_model.pkl"
    scaler_path = "scaler.pkl"
    feat_idx_path = "feature_indices.pkl"
    
    print("\nSelect the region for testing landslides:")
    print("1. Greece")
    print("2. Puerto Rico")
    region_choice = input("Enter 1 or 2: ").strip()
    
    if region_choice == "1":
        print("You selected Greece landslides.")
        shapefile_path = "/Users/deepaksingh/Documents/vscode/snehal/greece/ground_failure_polygons.shp"
        ground_failures = read_shapefiles_tda(shapefile_path)
        ground_failures = ground_failures[ground_failures['type'] != 'Liquefaction'].dropna(subset=['geometry'])
        test_inventory = ground_failures[ground_failures['epicentral'] == 'Greece'].reset_index(drop=True)
        inventory_name = "testt_region.tif"
    elif region_choice == "2":
        print("You selected Puerto Rico landslides.")
        shapefile_path = "/Users/deepaksingh/Documents/vscode/snehal/US_Landslide_v3_shp/us_ls_v3_poly.shp"
        test_inventoryy = read_shapefiles_tda(shapefile_path)
        test_inventoryyy = test_inventoryy[test_inventoryy['Inventory'] == 'USGS PR Hurricane Maria (Bessette-Kirton and others, 2019)']
        test_inventory = test_inventoryyy.reset_index(drop=True)
        inventory_name = "test_region.tif"
    else:
        print("Invalid selection. Exiting TDA-based classification.")
        return
    
    # For TDA, we use only the geometry column
    test_inventory = test_inventory[['geometry']]
    print(f"Number of polygons: {len(test_inventory)}")
    
    # Plot a sample polygon
    plot_polygon(test_inventory, polygon_index=0)
    
    # Set up DEM and point cloud file paths
    dem_location = "/Users/deepaksingh/Documents/vscode/snehal/dem"
    if not os.path.exists(dem_location):
        os.makedirs(dem_location)
    
    npz_pointcloud_file = f"pointcloud_test_region_{region_choice}.npz"
    features_excel = "features_test_region.xlsx"
    features_csv = "features_test_region.csv"
    
    dem_file = os.path.join(dem_location, inventory_name)
    if not os.path.exists(dem_file):
        download_dem(test_inventory, dem_location, inventory_name)
    
    # Create or load the 3D point cloud
    if os.path.exists(npz_pointcloud_file):
        npz_data = np.load(npz_pointcloud_file)
        pointcloud_test = [npz_data[f"arr_{i}"] for i in range(len(npz_data.files))]
    else:
        pointcloud_test = make_3d_polygons(test_inventory, dem_location, inventory_name, 1)
        np.savez_compressed(npz_pointcloud_file, *pointcloud_test)
    
    # Extract machine learning features from the 3D point cloud
    features_test = get_ml_features_with_files(pointcloud_test, output_excel=features_excel, output_csv=features_csv)
    print("Test features shape:", features_test.shape)
    
    if mode == "1":
        print("\nTraining new model using Japan training datasets...")
        japan_shapefiles = {
            "earthquake_hokkaido": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_hokkaido_polygons.shp",
            "earthquake_iwata": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_iwata_polygons.shp",
            "earthquake_niigata": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Earthquake_niigata_polygons.shp",
            "rainfall_kumamoto": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_kumamoto_polygons.shp",
            "rainfall_fukuoka": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_fukuoka_polygons.shp",
            "rainfall_saka": "/Users/deepaksingh/Documents/vscode/snehal/Japan Inventory/Rainfall_saka_polygons.shp",
        }
    
        japan_inventory_names = {
            "earthquake_hokkaido": "hokkaido.tif",
            "earthquake_iwata": "iwata.tif",
            "earthquake_niigata": "niigata.tif",
            "rainfall_kumamoto": "kumamoto.tif",
            "rainfall_fukuoka": "fukuoka.tif",
            "rainfall_saka": "saka.tif",
        }
    
        japan_pointclouds = {}
        japan_features = {}
    
        for key, shp in japan_shapefiles.items():
            print(f"Processing training data for {key}...")
            inventory_data = read_shapefiles_tda(shp)
            download_dem(inventory_data, dem_location, japan_inventory_names[key])
            japan_pointclouds[key] = make_3d_polygons(inventory_data, dem_location, japan_inventory_names[key], 1)
            japan_features[key] = get_ml_features_with_files(japan_pointclouds[key])
    
        eq_features = np.vstack((
            japan_features["earthquake_hokkaido"],
            japan_features["earthquake_iwata"],
            japan_features["earthquake_niigata"]
        ))
        rf_features = np.vstack((
            japan_features["rainfall_kumamoto"],
            japan_features["rainfall_fukuoka"],
            japan_features["rainfall_saka"]
        ))
    
        train_xgb_model(eq_features, rf_features, model_path, scaler_path, feat_idx_path)
    
    elif mode == "2":
        print("\nUsing pre-saved model for testing.")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feat_idx_path)):
            print("Saved model not found. Please train a model first.")
            return
    else:
        print("Invalid mode selection. Exiting.")
        return
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    indices = joblib.load(feat_idx_path)
    
    preds, pred_probs = predict_xgb_model(features_test, model, scaler, indices)
    
    number_rainfall_predicted = np.sum(preds)
    number_earthquake_predicted = len(preds) - number_rainfall_predicted
    probability_earthquake = round((number_earthquake_predicted / len(preds)) * 100, 4)
    probability_rainfall = round((number_rainfall_predicted / len(preds)) * 100, 4)
    
    print("\n--- Classification Results ---")
    print("Probability of inventory triggered by Earthquake:", f"{probability_earthquake}%")
    print("Probability of inventory triggered by Rainfall:", f"{probability_rainfall}%")
    print("TDA-based classification complete.")

# --------------------------------------------------
# Main: Let the user choose which classification method to run
# --------------------------------------------------
def main():
    print("Select classification method:")
    print("1. Image-Based Classification")
    print("2. Geometric-Based Classification")
    print("3. TDA-Based Classification")
    method = input("Enter 1, 2, or 3: ").strip()
    
    if method == "1":
        run_image_based_classification()
    elif method == "2":
        run_geometric_based_classification()
    elif method == "3":
        run_tda_based_classification()
    else:
        print("Invalid selection. Exiting.")

if __name__ == "__main__":
    main()
