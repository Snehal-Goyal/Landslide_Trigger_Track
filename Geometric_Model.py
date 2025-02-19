import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import shapely.affinity
from scipy.spatial import distance
from scipy.spatial import ConvexHull
import geopandas as gpd
import utm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import pandas as pd 
from shapely.geometry import Point, Polygon, MultiPoint
import shapely.geometry
from shapely.validation import explain_validity




def read_shapefiles (path_filename):
    
    """
    function to read the shapefile from the local file path of landslide inventory
    
  
    Parameters:
         :path_filename (str): path to local inventory shapefiles
    
    
    Returns:
         read shapefile from file path
    
    """
    
    return gpd.read_file(path_filename)



def latlon_to_eastnorth (lonlat_polydata):
    
    """ 
    function to convert the (longitude latitude) coordinates of polygons to (easting, northing) coordinates
    
    
    Parameters:
          :lonllat_polydata (array_like): 
                             longitude and latitude coordinates data
                      
    Returns:
            (array_like)
            easting and northing coordinates of landslide polygon data when polygon data has longitude latitude coordinates 

    
     
    """
     
    east_north_polydata=[]
    for i in range(np.shape(lonlat_polydata)[0]):
        u = utm.from_latlon(lonlat_polydata[i][1], lonlat_polydata[i][0])   ### (lat,lon) to (east,north)
        east_north_polydata.append([u[0],u[1]])
    east_north_polydata=np.asarray(east_north_polydata) 
    return  east_north_polydata  




def get_geometric_properties_landslide(poly_data):
    """
    Function to calculate the geometric properties of landslide polygons.

    Parameters:
        :poly_data: GeoPandas DataFrame containing landslide polygon geometries.

    Returns:
        (array_like) Geometric features of landslide polygons.
    """
    store_geometric_features_all_landslides = []    
    skipped_polygons = []  # To log skipped polygons and their reasons

    def check_polygon_validity(polygon, index):
        """
        Checks why a polygon is invalid and prints the reason.
        """
        if not polygon.is_valid:
            reason = explain_validity(polygon)
            print(f"Polygon at index {index} is invalid: {reason}")
            return reason
        if polygon.is_empty:
            print(f"Polygon at index {index} is empty.")
            return "Empty polygon"
        if polygon.area == 0:
            print(f"Polygon at index {index} has zero area.")
            return "Zero area"
        if len(polygon.exterior.coords) < 3:
            print(f"Polygon at index {index} has insufficient points to form a valid shape.")
            return "Insufficient points"
        return None

    for l in range((np.shape(poly_data)[0])):
        if poly_data['geometry'][l].geom_type != 'Polygon':
            skipped_polygons.append((l, "Non-polygon geometry"))
            continue

        z = np.asarray(poly_data['geometry'][l].exterior.coords)

        # Convert small coordinate values (e.g., lat/lon) to a projected system if needed
        if np.nanmin(z) < 100:
            z = latlon_to_eastnorth(z) 

        # Remove NaN values from the polygon coordinates
        ze = z[~np.isnan(z).any(axis=1)]

        # Ensure ze has enough points to form a valid polygon
        if len(ze) <= 2:
            skipped_polygons.append((l, "Insufficient points"))
            continue

        polygon = Polygon(ze)

        # Apply buffer(0) if the polygon is invalid
        if not polygon.is_valid:
            print(f"Attempting to fix invalid polygon at index {l} with buffer(0).")
            polygon = polygon.buffer(0)

        if polygon.is_empty or not polygon.is_valid:  # Check if still invalid
            reason = check_polygon_validity(polygon, l)
            skipped_polygons.append((l, reason if reason else "Unknown issue"))
            continue

        if polygon.area == 0:
            skipped_polygons.append((l, "Zero area polygon"))
            continue

        centre = np.array(polygon.centroid)

        if centre.size > 0:  # Ensure the centroid is valid
            # Calculate area and perimeter
            area_polygon = polygon.area
            perimeter_polygon = polygon.length

            # Fit minimum area bounding box
            bounding_box = MultiPoint(ze).minimum_rotated_rectangle
            coordinates_BB = np.asarray(bounding_box.exterior.coords)

            # Calculate the width of the bounding box and the rotation angle
            tan_y = (coordinates_BB[1, 1] - coordinates_BB[0, 1]) / (coordinates_BB[1, 0] - coordinates_BB[0, 0])
            dist_firstside = ((coordinates_BB[2, 1] - coordinates_BB[1, 1])**2 + (coordinates_BB[2, 0] - coordinates_BB[1, 0])**2)**0.5
            dist_secondside = ((coordinates_BB[1, 1] - coordinates_BB[0, 1])**2 + (coordinates_BB[1, 0] - coordinates_BB[0, 0])**2)**0.5
            width_minimum_boundingbox = min(dist_firstside, dist_secondside)
            theta = math.degrees(math.atan(tan_y))

            # Fit an ellipse to the polygon
            dw = np.sqrt((perimeter_polygon**4) - 16 * ((np.pi)**2) * (area_polygon**2))
            delta = (4 * np.pi * area_polygon) / ((perimeter_polygon**2) - dw)
            major_axis_ellipse = 2 * (np.sqrt((area_polygon * delta) / np.pi))
            minor_axis_ellipse = 2 * (np.sqrt((area_polygon) / (np.pi * delta)))

            # Calculate eccentricity of the ellipse
            eccentricity_ellipse = np.sqrt(1 - (minor_axis_ellipse**2) / (major_axis_ellipse**2))

            # Calculate convex hull and its measure
            hull = MultiPoint(ze).convex_hull
            convex_hull_area = hull.area
            convex_hull_measure = area_polygon / convex_hull_area

            # Store all features for the current polygon
            geometric_features_one_landslide = np.hstack((
                area_polygon,
                perimeter_polygon,
                area_polygon / perimeter_polygon,  # Compactness
                convex_hull_measure,
                minor_axis_ellipse,
                eccentricity_ellipse,
                width_minimum_boundingbox
            ))

            store_geometric_features_all_landslides.append(geometric_features_one_landslide)
        else:
            skipped_polygons.append((l, "Centroid calculation failed"))

    # Convert the stored features into a NumPy array for further analysis
    store_geometric_features_all_landslides = np.asarray(store_geometric_features_all_landslides)
    print(f"Processed {len(store_geometric_features_all_landslides)} valid polygons.")
    print(f"Skipped polygons: {len(skipped_polygons)}")
    for index, reason in skipped_polygons:
        print(f"Polygon at index {index} was skipped: {reason}")

    return store_geometric_features_all_landslides



def classify_inventory_xgb(earthquake_inventory_features, rainfall_inventory_features, test_inventory_features):
    """
    Function to predict the trigger of landslides in testing inventory using predefined best hyperparameters for XGBoost.

    Parameters:
       :earthquake_inventory_features (array_like): geometric features of known earthquake inventories landslides
       :rainfall_inventory_features (array_like): geometric features of known rainfall inventories landslides
       :test_inventory_features (array_like): geometric features of known testing inventory landslides  

    Returns:
        (array_like) probability of testing inventory landslides belonging to earthquake and rainfall class
    """

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Best hyperparameters from Optuna
    best_params = {
        'n_estimators': 206,
        'max_depth': 8,
        'learning_rate': 0.11516658376780635,
        'subsample': 0.5935295674635599,
        'colsample_bytree': 0.7324694050847117,
        'gamma': 3.186905646175651,
        'reg_alpha': 0.4112601666885729,
        'reg_lambda': 1.0210808671242577,
        'random_state': 42
    }

    earthquake_label = np.zeros((np.shape(earthquake_inventory_features)[0], 1))
    rainfall_label = np.ones((np.shape(rainfall_inventory_features)[0], 1))

    n1 = np.shape(earthquake_inventory_features)[0]
    n2 = np.shape(rainfall_inventory_features)[0]

    if n1 > n2:
        indi_earth = random.sample(range(n1), n2)
        train_earth = earthquake_inventory_features[indi_earth, :]
        train_label_earth = earthquake_label[indi_earth]
        train_rain = rainfall_inventory_features
        train_label_rain = rainfall_label
    else:
        indi_rain = random.sample(range(n2), n1)
        train_rain = rainfall_inventory_features[indi_rain, :]
        train_label_rain = rainfall_label[indi_rain]
        train_earth = earthquake_inventory_features
        train_label_earth = earthquake_label

    train_data = np.vstack((train_earth, train_rain))
    train_label = np.vstack((train_label_earth, train_label_rain))

    # Standardize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_inventory_features)

    # Initialize XGBoost classifier with best hyperparameters
    clf = XGBClassifier(**best_params)
    clf.fit(train_data, np.ravel(train_label))

    # Predict probabilities
    predictions = clf.predict_proba(test_data)
    y_pred = clf.predict(test_data)

    number_rainfall_predicted_landslides = np.sum(y_pred)
    number_earthquake_predicted_landslides = np.shape(y_pred)[0] - number_rainfall_predicted_landslides

    probability_earthquake_triggered_inventory = (number_earthquake_predicted_landslides / np.shape(y_pred)[0]) 
    probability_rainfall_triggered_inventory = (number_rainfall_predicted_landslides / np.shape(y_pred)[0]) 

    probability_earthquake_triggered_inventory = np.round(probability_earthquake_triggered_inventory, 4)
    probability_rainfall_triggered_inventory = np.round(probability_rainfall_triggered_inventory, 4)

    print(f"Probability of inventory triggered by Earthquake: {probability_earthquake_triggered_inventory}%")
    print(f"Probability of inventory triggered by Rainfall: {probability_rainfall_triggered_inventory}%")

    return predictions

    
def plot_geometric_results(predict_proba):
    
    
    """
    function to visualize the trigger prediction of landslides in testing inventory
    
     
    Parameters:
         :predict_proba (array_like): probability of each landslide in inventory class belonging to earthquake and rainfall class.
                   
                   
    Returns:
         Visualization of landslide probabilities belong to earthquake and rainfall class and trigger prediction of entire landslide 
         inventory 
                
    """
    
    plt.rc('text', usetex=True)
    # chage xtick and ytick fontsize 
    # chage xtick and ytick fontsize 
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    n1=np.shape(np.argwhere(predict_proba[:,0]>0.5))[0]
    n2=np.shape(np.argwhere(predict_proba[:,1]>0.5))[0]
    
    def RF_image(predict_proba):
        predict_proba=np.int32(np.round(predict_proba*100))
        data=np.zeros((np.shape(predict_proba)[0],100))
        for i in range(np.shape(predict_proba)[0]):
            a,b=predict_proba[i,0],predict_proba[i,1]
            #################
            c=np.zeros(int(a),)
            d=np.ones(int(b),)
            if int(a)==100:
                mat=c
            elif int(b)==100:
                mat=d
            else:   
                mat=np.hstack((c,d))
            data[i,:]=mat
        data=np.transpose(data)
        return data 

    import matplotlib as mpl
    fig,ax=plt.subplots(1, 1,figsize=(14,6), constrained_layout=True)
    cm = mpl.colors.ListedColormap([[230/255,204/255,179/255],[30/255,100/255,185/255]])
    
    if n1>n2: 
        earthquake_accuracy=np.round((n1/(n1+n2))*100,2) 
        ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Earthquake: %s  '%earthquake_accuracy,fontsize=26)
        
        ind=np.argsort(predict_proba[:,0])
        predict_proba=predict_proba[ind,:]        
        matrix_probability=RF_image(predict_proba)
        image=np.dstack((matrix_probability,matrix_probability,matrix_probability))
        image[matrix_probability[:,:]==0]=[230,204,179]
        image[matrix_probability[:,:]==1]=[30,100,185]
        image=np.int32(image)
        
        
    else:
        rainfall_accuracy=np.round((n2/(n1+n2))*100,2) 
        ind=np.argsort(predict_proba[:,1])
        predict_proba=predict_proba[ind,:]        
        matrix_probability=RF_image(predict_proba)
        image=np.dstack((matrix_probability,matrix_probability,matrix_probability))
        image[matrix_probability[:,:]==0]=[230,204,179]
        image[matrix_probability[:,:]==1]=[30,100,185]
        image=np.int32(image)
        ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Rainfall: %s '%rainfall_accuracy+'%',fontsize=26)
        #ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Rainfall: %s '%rainfall_accuracy,fontsize=26)

        image=np.flipud(image)

    
    
    
    pcm = ax.imshow(image,aspect='auto',cmap=cm,origin='lower')
    ax.set_xlabel('Test Sample Index',fontsize=26)
    ax.set_ylabel('Class Probability',fontsize=26)
    

    #ax.text(1380,110,'85.31 $\pm$ 0.19 \%',fontsize=26)


    cb=plt.colorbar(pcm, location='top',pad=0.03,ax=ax)
    cb.ax.set_xticklabels([],length=0)                 # vertically oriented colorbar
    cb.ax.set_yticklabels([],length=0)                 # vertically oriented colorbar
    cb.ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.text(np.shape(predict_proba)[0]//6,135,'Earthquake',fontsize=26)
    ax.text(np.shape(predict_proba)[0]//1.4,135,'Rainfall',fontsize=26)

    #cb.set_label('Earthquake                            Rainfall ',fontsize=26)
    plt.show()
 


def plot_bar_chart(predict_proba):
    """
    Plots a bar chart showing the probability of landslides being triggered by earthquakes and rainfall.
    """
    earthquake_probs = predict_proba[:, 0] * 100
    rainfall_probs = predict_proba[:, 1] * 100

    # Sorting for better visualization
    sorted_indices = np.argsort(earthquake_probs)
    earthquake_probs = earthquake_probs[sorted_indices]
    rainfall_probs = rainfall_probs[sorted_indices]

    x = np.arange(len(earthquake_probs))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, earthquake_probs, label="Earthquake Trigger Probability", color='orange', alpha=0.7)
    ax.bar(x, rainfall_probs, label="Rainfall Trigger Probability", color='blue', alpha=0.7, bottom=earthquake_probs)

    ax.set_xlabel("Landslide Sample Index")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Landslide Trigger Classification - Bar Chart")
    ax.legend()
    plt.show()

def plot_area_chart(predict_proba):
    """
    Plots a stacked area chart for landslide trigger probability distribution.
    """
    earthquake_probs = predict_proba[:, 0] * 100
    rainfall_probs = predict_proba[:, 1] * 100

    # Sorting for better visualization
    sorted_indices = np.argsort(earthquake_probs)
    earthquake_probs = earthquake_probs[sorted_indices]
    rainfall_probs = rainfall_probs[sorted_indices]

    x = np.arange(len(earthquake_probs))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, 0, earthquake_probs, color='orange', alpha=0.6, label="Earthquake Trigger Probability")
    ax.fill_between(x, earthquake_probs, earthquake_probs + rainfall_probs, color='blue', alpha=0.6, label="Rainfall Trigger Probability")

    ax.set_xlabel("Landslide Sample Index")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Landslide Trigger Probability Distribution - Area Chart")
    ax.legend()
    plt.show()

def plot_scatter_chart(predict_proba):
    """
    Plots a scatter chart where each landslide sample is color-coded based on the higher trigger probability.
    """
    x = np.arange(len(predict_proba))
    y = np.zeros(len(predict_proba))  # Dummy y-values for visualization
    colors = np.where(predict_proba[:, 0] > 0.5, 'orange', 'blue')

    fig, ax = plt.subplots(figsize=(12, 4))
    scatter = ax.scatter(x, y, c=colors, alpha=0.6, edgecolors='k')

    ax.set_xlabel("Landslide Sample Index")
    ax.set_title("Landslide Trigger Classification: Earthquake vs. Rainfall - Scatter Chart")
    ax.yaxis.set_visible(False)  # Hide Y-axis as it's not needed

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label="Earthquake"),
                       plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label="Rainfall")]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.show()


def visualize_all(predict_proba):
    """
    Calls all visualization functions in sequence to compare different visual representations.
    """
    print("Displaying Bar Chart...")
    plot_bar_chart(predict_proba)

    print("Displaying Area Chart...")
    plot_area_chart(predict_proba)

    print("Displaying Scatter Chart...")
    plot_scatter_chart(predict_proba)




 


