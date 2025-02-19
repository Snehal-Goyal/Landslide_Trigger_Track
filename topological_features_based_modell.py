##  Python libraries needed to run the TDA based method package ##

import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import MultiPoint
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
from sklearn.ensemble import RandomForestClassifier
import time
#import gdal
from pyproj import Proj,transform
import elevation
from osgeo import gdal

import pandas as pd
from scipy.interpolate import griddata

from gtda.plotting import plot_diagram
from gtda.homology import VietorisRipsPersistence,SparseRipsPersistence,EuclideanCechPersistence
from gtda.diagrams import Amplitude,NumberOfPoints,PersistenceEntropy
from gtda.diagrams import Filtering
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split




def read_shapefiles (path_filename):
    
    """
    function to read the shapefile from the local file path of landslide inventory
    
  
    Parameters:
         :path_filename (str): path to local inventory shapefiles
    
    
    Returns:
         read shapefile from file path
    
    """
    
    return gpd.read_file(path_filename)

def min_max_inventory(poly_data,lon_res,lat_res):

    """
    function to calculate the bounding box coordinates of complete landslide inventory


    Parameters:
          :poly_data (str): landslide polygon data in an inventory
          :lon_res (float): longitude resolution
          :lat_res (float): latitude resolution

    
    Returns:
         bounding box coordinates of landslide inventory region
    
    """
    data_coord=[]
    for l in range((np.shape(poly_data)[0])):
        if poly_data['geometry'][l].geom_type=='Polygon':
            poly_xy=np.asarray(poly_data['geometry'][l].exterior.coords)  ## (lon,lat)
            min_landslide_lon,max_landslide_lon=np.min(poly_xy[:,0]),np.max(poly_xy[:,0])
            min_landslide_lat,max_landslide_lat=np.min(poly_xy[:,1]),np.max(poly_xy[:,1])
            data_coord.append([min_landslide_lon,max_landslide_lon,min_landslide_lat,max_landslide_lat])
    data_coord=np.asarray(data_coord) 
    kk=20
    
    return (np.min(data_coord[:,0])-kk*lon_res, np.max(data_coord[:,1])+kk*lon_res,np.min(data_coord[:,2])+kk*lat_res,np.max(data_coord[:,3])-kk*lat_res)


def download_dem(poly_data,dem_location,inventory_name):

    """
    function to download the DEM corresponding to inventory region

    Parameters:
         :poly_data (str) : landslide polygon data in an inventory
         :dem_location (str): provide the path where user wants to download DEM
         :inventory_name (str): inventory_name to save the dem file

    Returns:
        (str) downloaded DEM file location for input landslide inventory
          
    """
    
    longitude_min,longitude_max,latitude_min,latitude_max=min_max_inventory(poly_data,0.00,-0.00)

    total_number_of_tiles=(longitude_max-longitude_min)*(latitude_max-latitude_min)
    print('total number of tiles:',total_number_of_tiles)
    print("** Number of tiles should be less than 100 or depend on user device RAM **" )
    print('** only the folder location in dem_location option **')

    
    #inventory_name=input('**only tif name should be given')
    #inventory_name='inventory%s'%np.random.randint(0,1000)+'.tif'
    final_output_filename=dem_location+inventory_name
    if total_number_of_tiles<10:
       longitude_min,longitude_max=longitude_min-0.4,longitude_max+0.4
       latitude_min,latitude_max=latitude_min-0.4,latitude_max+0.4
       latitude_min,latitude_max
       print("less than 10 tiles") 
       elevation.clip(bounds=(longitude_min, latitude_min, longitude_max, latitude_max), output=final_output_filename)
       elevation.clean() 

    else:
        print('more than 10 tiles')
        latitude_width=int(latitude_max-latitude_min)
        longitude_width=int(longitude_max-longitude_min)

        add_latitude=3-latitude_width%3
        add_longitude=3-longitude_width%3

        latitude_max=latitude_max+add_latitude
        longitude_max=longitude_max+add_longitude

        latitude_width=(latitude_max-latitude_min)
        longitude_width=(longitude_max-longitude_min)
        t=0
        for j in range(0,latitude_width,3):
            for i in range(0,longitude_width,3):
                t=t+1
                output=dem_location+'inven_name%s.tif'%t
                elevation.clip(bounds=(longitude_min+i, latitude_max-j-3, longitude_min+i+3,latitude_max-j), output=output)    
                elevation.clean()

        NN=10800
        DEM_DATA=np.zeros((NN*latitude_width//3, NN*longitude_width//3),dtype='uint16')
        t=1
        X_0,Y_0=[],[]


        for i in range(latitude_width//3):
            for j in range(longitude_width//3):
                inv_name="inven_name%s.tif"%t
                data_name=dem_location+inv_name
                DEM = gdal.Open(data_name)
                x_0,x_res,_,y_0,_,y_res = DEM.GetGeoTransform()
                X_0.append(x_0),Y_0.append(y_0)
                print(x_0,x_res,_,y_0,_,y_res)
                #print(np.asarray(DEM))
                from PIL import Image
                #im = Image.open(data_name)
                #z = np.array(DEM.GetRasterBand().ReadAsArray())

                z=gdal.Dataset.ReadAsArray(DEM)
                DEM_DATA[(i*NN):(i*NN)+NN,(j*NN):(j*NN)+NN]=z
                t=t+1
                print(t)
        x_0=min(X_0)
        y_0=max(Y_0)
        time.sleep(180)
        #######################################################################################################
        geotransform = (x_0,x_res,0,y_0,0,y_res)
        driver = gdal.GetDriverByName('Gtiff')
        final_output_filename=dem_location+inventory_name
        dataset = driver.Create(final_output_filename, DEM_DATA.shape[1], DEM_DATA.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotransform)
        dataset.GetRasterBand(1).WriteArray(DEM_DATA)
        #################################################################################################    
    time.sleep(180)
    return  final_output_filename 


def make_3d_polygons(poly_data,dem_location,inventory_name,kk):

    """    
    function to get 3D point cloud from 2D shape of landslide

    Parameters:
       :poly_data (str): polygons shapefile
       :dem_location (str): path of dem file
       :inventory_name (str): path of dem file
       :kk (int): kk=1 if user have already DEM corresponding to inventory region otherwise use any other number
   
    Returns:
       (array_like) 3D data of landslides
       
    """
    
    if kk==1:  
       DEM_FILE_NAME=dem_location+inventory_name
    else:
         DEM_FILE_NAME=download_dem(poly_data,dem_location,inventory_name)
    ############################################################################
    inProj = Proj(init='epsg:4326') #Proj(init='epsg:4326') creates a projection object based on the EPSG:4326 standard
    #The EPSG:4326 code represents the World Geodetic System 1984 (WGS84), which is a widely 
    #used global coordinate system for representing locations on the Earth using latitude and longitude in degrees.
    
    outProj = Proj(init='epsg:3857') 
    # EPSG:3857 is a projected coordinate system (often used for web mapping)
    # where the Earth is projected onto a flat map, and distances are measured in meters.
    data=[]

    DEM_FILE_NAME = os.path.join(dem_location, inventory_name)

    # ✅ Check if the DEM file exists before trying to open it
    if not os.path.exists(DEM_FILE_NAME):
        raise FileNotFoundError(f"DEM file not found: {DEM_FILE_NAME}. Please check the download process.")

    DEM = gdal.Open(DEM_FILE_NAME)
    
    # ✅ Check if gdal.Open() failed
    if DEM is None:
        raise ValueError(f"Failed to load DEM file: {DEM_FILE_NAME}. The file may be corrupted or unreadable.")


    #poen dem file using gdal library
    DEM = gdal.Open(DEM_FILE_NAME)

    # GetGeoTransform(): Extracts the DEM's geotransform information
    # lon_init: Starting longitude.
    # lon_res: Resolution or pixel size in the longitude direction.
    # lat_init: Starting latitude.
    # lat_res: Resolution in the latitude direction.
    lon_init,lon_res,_,lat_init,_,lat_res = DEM.GetGeoTransform()
    DEM_data=gdal.Dataset.ReadAsArray(DEM)
    #print(np.shape(DEM_data))

    lon_all=np.arange(lon_init,lon_init+np.shape(DEM_data)[1]*lon_res,lon_res)
    # An array of longitudes corresponding to each column in the DEM grid, 
    # starting from the upper-left corner and extending to the right.

    lat_all=np.arange(lat_init,lat_init+np.shape(DEM_data)[0]*lat_res,lat_res)
    # An array of latitudes corresponding to each row in the DEM grid, starting
    #  from the upper-left corner and extending downward.


    #print (' ***  Upload Complete Shapefiles Of Landslides In Landslide Inventory ***')
    #print('*** Input should be a shapefiles of landslide polygons  *** ' )
    
    inv_lon_min,inv_lon_max,inv_lat_min,inv_lat_max=min_max_inventory(poly_data,lon_res,lat_res)
    # his function (assumed to be defined elsewhere) likely computes the minimum and maximum longitude 
    # and latitude values for the area represented by poly_data, which may contain geometric shapes (like polygons) 
    # defining certain regions of interest.
    #inv_lon_min: The minimum longitude of the inventory area.
    #inv_lon_max: The maximum longitude of the inventory area.
    #inv_lat_min: The minimum latitude of the inventory area.
    #inv_lat_max: The maximum latitude of the inventory area.


    indices_lon_dem_crop_inventory=np.argwhere((lon_all>inv_lon_min)&(lon_all<inv_lon_max))[:,0]
    # indices_lon_dem_crop_inventory: This array contains the indices of the longitudes in lon_all 
    # that are within the specified inventory longitude range

    indices_lat_dem_crop_inventory=np.argwhere((lat_all>inv_lat_min)&(lat_all<inv_lat_max))[:,0]
    # indices_lat_dem_crop_inventory: This array contains the indices of the latitudes in lat_all 
    # that are within the specified inventory latitude range.

    min_indices_lon_dem_crop_inventory=np.min(indices_lon_dem_crop_inventory)
   #min_indices_lon_dem_crop_inventory: The minimum index for longitude that will be analyzed.


    max_indices_lon_dem_crop_inventory=np.max(indices_lon_dem_crop_inventory)
         #max_indices_lon_dem_crop_inventory: The maximum index for longitude that will be analyzed.

    min_indices_lat_dem_crop_inventory=np.min(indices_lat_dem_crop_inventory)
            #min_indices_lat_dem_crop_inventory: The minimum index for latitude that will be analyzed.

    max_indices_lat_dem_crop_inventory=np.max(indices_lat_dem_crop_inventory)
         # max_indices_lat_dem_crop_inventory: The maximum index for latitude that will be analyzed.



    DEM_data=DEM_data[min_indices_lat_dem_crop_inventory:max_indices_lat_dem_crop_inventory,
                          min_indices_lon_dem_crop_inventory:max_indices_lon_dem_crop_inventory]
    #  this line crops the DEM_data array to include only the relevant section defined by the minimum
    #  and maximum latitude and longitude indices calculated previously.

    
    lon_all=lon_all[min_indices_lon_dem_crop_inventory:max_indices_lon_dem_crop_inventory]

    lat_all=lat_all[min_indices_lat_dem_crop_inventory:max_indices_lat_dem_crop_inventory]           
    
    for l in range((np.shape(poly_data)[0])):
    #for l in range(300):    
        if poly_data['geometry'][l].geom_type=='Polygon':  #Only processes geometries of type 'Polygon'.
            #print(l)
            poly_xy=np.asarray(poly_data['geometry'][l].exterior.coords)  ## (lon,lat)
            #Converts the exterior coordinates of the polygon to a NumPy array called poly_xy,
            #  where the first column represents longitudes and the second represents latitudes.

           # Calculates the minimum and maximum longitudes and latitudes for the current polygon,
           #  which defines the geographic bounds of the landslide area.
            min_landslide_lon,max_landslide_lon=np.min(poly_xy[:,0]),np.max(poly_xy[:,0])
            min_landslide_lat,max_landslide_lat=np.min(poly_xy[:,1]),np.max(poly_xy[:,1])



            extra=10 # An additional buffer of 10 pixels is added to ensure coverage of the landslide area in the DEM data.
            indices_lon_land=np.argwhere((lon_all>min_landslide_lon-extra*lon_res) & (lon_all<max_landslide_lon+extra*lon_res))[:,0]
            #indices_lon_land finds the indices in the lon_all array that are within the adjusted minimum and maximum longitudes.
            #indices_lat_land finds the indices in the lat_all array within the adjusted minimum and maximum latitudes.
            indices_lat_land=np.argwhere((lat_all>min_landslide_lat+extra*lat_res) & (lat_all<max_landslide_lat-extra*lat_res))[:,0]
            
            
            
            #Extracts a sub-region of the DEM data that corresponds to the bounds determined from the polygon's coordinates.
            DEM_landslide_region_crop=DEM_data[np.min(indices_lat_land):np.max(indices_lat_land)+1,
                                              np.min(indices_lon_land):np.max(indices_lon_land)+1] ############## check 
            
            #Longitude and Latitude Arrays: Extracts the relevant longitudes and latitudes for the cropped DEM.
            lon_landslides_region=lon_all[indices_lon_land]
            lat_landslides_region=lat_all[indices_lat_land]

            ######## for landslide region interpolation #######
            lon_mesh,lat_mesh=np.meshgrid(lon_landslides_region,lat_landslides_region)
            lon_mesh,lat_mesh=lon_mesh.flatten(),lat_mesh.flatten()
            DEM_landslide_region_crop_=DEM_landslide_region_crop.flatten()

            
            #  Coordinate Transformation: Transforms the longitude and latitude coordinates from one projection to another
            #  using the specified input (inProj) and output (outProj) projections. This is crucial for ensuring spatial alignment.
            lon_mesh_east,lat_mesh_north = transform(inProj,outProj,lon_mesh,lat_mesh)

            poly_xy[:,0],poly_xy[:,1] = transform(inProj,outProj,poly_xy[:,0],poly_xy[:,1])

            
            lon_mesh_east=np.reshape(lon_mesh_east,(np.shape(lon_mesh_east)[0],1))
            lat_mesh_north=np.reshape(lat_mesh_north,(np.shape(lat_mesh_north)[0],1))
            lonlat_mesh_eastnorth=np.hstack((lon_mesh_east,lat_mesh_north))
            

            # Bounding Box Adjustment: Defines a bounding box around the polygon by adding a buffer of 30 units to each side.
            xmin1,xmax1=np.min(poly_xy[:,0])-30,np.max(poly_xy[:,0])+30
            ymin1,ymax1=np.min(poly_xy[:,1])-30,np.max(poly_xy[:,1])+30
            k,total_grid=0,32
            xnew =np.linspace(xmin1-k, xmax1+k,total_grid)
            ynew =np.linspace(ymin1-k, ymax1+k,total_grid) 
            
            # Generates a new grid (xnew, ynew) for interpolation that spans the adjusted bounding box, with total_grid points in each directio
            xneww,yneww=np.meshgrid(xnew,ynew)
           
             
            eleva_inter=griddata(lonlat_mesh_eastnorth, DEM_landslide_region_crop_,(xneww,yneww),method='cubic')
            # Uses the griddata function to interpolate the DEM data onto the new grid defined by xnew and ynew. 
            ####The method='cubic' argument specifies the type of interpolation###
    
            eleva_final=eleva_inter
            eleva_norm=(eleva_final-np.min(eleva_final))/(np.max(eleva_final)-np.min(eleva_final))
               #he interpolated elevation data is stored in eleva_final.

            #######################################################################################################################
            #Preparing Data for Output
            polygon = Polygon(poly_xy)
            XNEW,YNEW=np.meshgrid(xnew,ynew)
            XNEW,YNEW=XNEW.flatten(),YNEW.flatten()
            combine_data=np.zeros((total_grid*total_grid,3))
            combine_data[:,0]=XNEW
            combine_data[:,1]=YNEW

              #print('elevation')
            ELEVA_NORM=eleva_norm.flatten()
            combine_data[:,2]=ELEVA_NORM
  
            ##################################################################################################
            
            #Loops through each point in combine_data to check if it lies within the polygon. If it does, its index is added to the indices list.
            indices=[]
            for i in range(np.shape(combine_data)[0]):
                point=Point(combine_data[i,0:2])
                if polygon.contains(point)==True:
                   indices.append(i) 
            # Final Data Adjustment and Return
            indices=np.asarray(indices)
            if np.shape(indices)[0]>0:
                combine_data=combine_data[indices]
                combine_data[:,0]=combine_data[:,0]-np.min(combine_data[:,0])
                combine_data[:,1]=combine_data[:,1]-np.min(combine_data[:,1])
                 #Filtering: If any indices were found, combine_data is filtered to retain only the points within the polygon.
                 #Re-centering: The coordinates are re-centered by subtracting the minimum values to start from (0, 0).
                 # Appending Results: The filtered and adjusted data is added to the data list for eventual output.
                 # Finally, the function returns the data, which contains the processed elevation and coordinate information for all the polygons.                 
                data.append(combine_data)
    return data



def get_ml_features_with_files(data, output_excel='data.xlsx', output_csv='data.csv'):
    """
    Function to get machine learning features from 3D point cloud data and save to Excel and CSV.

    Parameters:
         :data (list): List of 3D point cloud arrays
         :output_excel (str): Path to save the Excel file with features
         :output_csv (str): Path to save the CSV file with features
   
    Returns:
          Topological features corresponding to 3D point cloud data (array)
    """
    if not isinstance(data, list) or not all(isinstance(arr, np.ndarray) for arr in data):
        raise ValueError("Input data must be a list of 3D point cloud arrays (NumPy arrays).")
    
    # Generate default polygon names
    polygon_names = [f"Polygon_{i}" for i in range(len(data))]
    data_name = [k for k, v in locals().items() if v is data][0] if any(v is data for v in locals().values()) else "data_features"
    if output_excel is None:
        output_excel = f"{data_name}.xlsx"
    if output_csv is None:
        output_csv = f"{data_name}.csv"

    homology_dimensions = [0, 1, 2]
    persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6, collapse_edges=True)
    persistence_diagrams = persistence.fit_transform(data)
    
    def average_lifetime(pers_diagrams_one):
        persistence_table = pd.DataFrame(pers_diagrams_one, columns=["birth", "death", "homology_dim"])
        persistence_table["lifetime"] = persistence_table["death"] - persistence_table["birth"]
        life_avg_all_dims = []
        for dims in homology_dimensions:
            avg_lifetime_one = persistence_table[persistence_table['homology_dim'] == dims]['lifetime'].mean()
            life_avg_all_dims.append(avg_lifetime_one)
        return np.asarray(life_avg_all_dims).flatten()

    metrics = ["bottleneck", "wasserstein", "landscape", 'heat', 'betti', "silhouette"]
    feature_all_data = []
    column_names = ["Polygon Name"]  # Adding columns for polygon names

    for dim in homology_dimensions:
        column_names.append(f"Persistence Entropy H{dim}")
    for dim in homology_dimensions:
        column_names.append(f"Number of Points H{dim}")
    for dim in homology_dimensions:
        column_names.append(f"Average Lifetimes H{dim}")
    for metric in metrics:
        for dim in homology_dimensions:
            column_names.append(f"Amplitude ({metric}) H{dim}")

    for i in range(len(persistence_diagrams)):
        feature_total_one = [polygon_names[i]]  # Add polygon name
        persistant_one = persistence_diagrams[i][None, :, :]

        # Persistence Entropy
        persistence_entropy = PersistenceEntropy()
        feature_total_one.extend(persistence_entropy.fit_transform(persistant_one).flatten())

        # Number of Points
        feature_total_one.extend(NumberOfPoints().fit_transform(persistant_one).flatten())

        # Average Lifetime
        feature_total_one.extend(average_lifetime(persistence_diagrams[i]))

        # Amplitudes
        for metric in metrics:
            feature_total_one.extend(Amplitude(metric=metric).fit_transform(persistant_one).flatten())

        feature_all_data.append(feature_total_one)

    feature_all_data = np.asarray(feature_all_data, dtype=object)

    # Save features to Excel and CSV
    features_df = pd.DataFrame(feature_all_data, columns=column_names)
    features_df.to_excel(output_excel, index=False)
    features_df.to_csv(output_csv, index=False)

    return feature_all_data
# you will get excel and csv files of features of the data




def classify_inventory_tda_with_xgboost(
    earthquake_inventory_features, rainfall_inventory_features, test_inventory_features
):
    """
    Function to classify testing inventory landslides as triggered by earthquakes or rainfall 
    using XGBoost while addressing class imbalance.

    Parameters:
        :earthquake_inventory_features (array_like): TDA features of known earthquake inventories landslides
        :rainfall_inventory_features (array_like): TDA features of known rainfall inventories landslides
        :test_inventory_features (array_like): TDA features of testing inventory landslides    

    Returns:
        predictions (array_like): Probabilities of testing inventory landslides belonging to earthquake and rainfall classes
    """
    # Remove the first column (polygon names) if it contains non-numeric data
    try:
        if isinstance(earthquake_inventory_features[0, 0], str):
            earthquake_inventory_features = earthquake_inventory_features[:, 1:]
        if isinstance(rainfall_inventory_features[0, 0], str):
            rainfall_inventory_features = rainfall_inventory_features[:, 1:]
        if isinstance(test_inventory_features[0, 0], str):
            test_inventory_features = test_inventory_features[:, 1:]

    except IndexError as e:
        raise ValueError("Input data must have at least one feature column.") from e

    # Convert features to float
    try:
        earthquake_inventory_features = earthquake_inventory_features.astype(float)
        rainfall_inventory_features = rainfall_inventory_features.astype(float)
        test_inventory_features = test_inventory_features.astype(float)
    except ValueError as e:
        raise ValueError("Non-numeric values detected in the feature columns.") from e
    

    # Labels for the two classes
    earthquake_label = np.zeros((np.shape(earthquake_inventory_features)[0], 1))
    rainfall_label = np.ones((np.shape(rainfall_inventory_features)[0], 1))

    # Handle class imbalance by undersampling or oversampling
    n1 = np.shape(earthquake_inventory_features)[0]  # Number of earthquake samples
    n2 = np.shape(rainfall_inventory_features)[0]  # Number of rainfall samples

    if n1 > n2:  # More earthquake samples
        indi_earth = random.sample(range(n1), n2)  # Randomly sample n2 earthquake instances
        train_earth = earthquake_inventory_features[indi_earth, :]
        train_label_earth = earthquake_label[indi_earth]
        train_rain = rainfall_inventory_features
        train_label_rain = rainfall_label
    else:  # More rainfall samples
        indi_rain = random.sample(range(n2), n1)  # Randomly sample n1 rainfall instances
        train_rain = rainfall_inventory_features[indi_rain, :]
        train_label_rain = rainfall_label[indi_rain]
        train_earth = earthquake_inventory_features
        train_label_earth = earthquake_label

    # Combine the balanced data
    train_data = np.vstack((train_earth, train_rain))
    train_label = np.vstack((train_label_earth, train_label_rain))

    # Scale the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_inventory_features)

    # Best parameters obtained from Optuna
    best_params = {
        "max_depth": 3,  # Replace with your best max_depth
        "learning_rate": 0.1,#0.02456,  # Replace with your best learning_rate
        "n_estimators": 650,  # Replace with your best n_estimators
        "gamma": 20,  # Replace with your best gamma
        "subsample": 0.7404,  # Replace with your best subsampleo
        "colsample_bytree": 0.9498,  # Replace with your best colsample_bytree
        "alpha":5,
        "lambda":5,
        "min_child_weight":5,
        "scale_pos_weight": 0.63626821
    }

    # Train the XGBoost model
    clf = XGBClassifier(**best_params, eval_metric="logloss")
    clf.fit(train_data, train_label.ravel())

    # Feature importance for selection
    feature_importance = clf.feature_importances_
    indices = np.argsort(-feature_importance)[:10]  # Select top 10 important features

    # Train the final model with top features
    classifier = XGBClassifier(**best_params, eval_metric="logloss")
    classifier.fit(train_data[:, indices], train_label.ravel())

    # Make predictions on test data
    y_pred = classifier.predict(test_data[:, indices])
    predictions = classifier.predict_proba(test_data[:, indices])


    

    # Calculate class probabilities
    number_rainfall_predicted = np.sum(y_pred)
    number_earthquake_predicted = np.shape(y_pred)[0] - number_rainfall_predicted

    probability_earthquake_triggered_inventory = (
        number_earthquake_predicted / np.shape(y_pred)[0]
    ) * 100
    probability_rainfall_triggered_inventory = (
        number_rainfall_predicted / np.shape(y_pred)[0]
    ) * 100

    probability_earthquake_triggered_inventory = np.round(
        probability_earthquake_triggered_inventory, 2
    )
    probability_rainfall_triggered_inventory = np.round(
        probability_rainfall_triggered_inventory, 2
    )

    print(
        "Probability of inventory triggered by Earthquake: ",
        str(float(probability_earthquake_triggered_inventory)) + "%",
    )
    print(
        "Probability of inventory triggered by Rainfall: ",
        str(float(probability_rainfall_triggered_inventory)) + "%",
    )

    return predictions


def classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features):
    
    """
    function to give probability of testing inventory belonging to earthquake and rainfall class
     
    Parameters:
        :earthquake_inventory_features (array_like): TDA features of known earthquake inventories landslides
        :rainfall_inventory_features (array_like):  TDA features of known rainfall inventories landslides
        :test_inventory_features (array_like): TDA features of known testing inventory landslides    

    Returns:
           (array_like) probability of testing inventory landslides belonging to earthquake and rainfall class

    """
    
    earthquake_label=np.zeros((np.shape(earthquake_inventory_features)[0],1))
    rainfall_label=np.ones((np.shape(rainfall_inventory_features)[0],1))
    
    n1=np.shape(earthquake_inventory_features)[0]
    n2=np.shape(rainfall_inventory_features)[0]
    if n1>n2:  ### n1 is number of earth samples and n2 is number of rainfall samples #####
        indi_earth=random.sample(range(n1),n2)
        train_earth=earthquake_inventory_features[indi_earth,:]
        train_label_earth=earthquake_label[indi_earth]  ## checked
        train_rain=rainfall_inventory_features
        train_label_rain=rainfall_label
        #######################################################################################
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

    else:
        indi_rain=random.sample(range(n2),n1)
        train_rain=rainfall_inventory_features[indi_rain,:]
        train_label_rain=rainfall_label[indi_rain]   
        train_earth=earthquake_inventory_features
        train_label_earth=earthquake_label        
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

        #########################################################################################
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    
    test_data=test_inventory_features
    test_data = scaler.transform(test_data)    
    clf=RandomForestClassifier(n_estimators=1000,max_depth=5)
    clf.fit(train_data,np.ravel(train_label))
    
    ############ feature importance selections ######
    Featur_importance=clf.feature_importances_
    indices=np.argsort(-Featur_importance)[0:10]
    
    classifier=RandomForestClassifier(n_estimators=1000,max_depth=5)
    classifier.fit(train_data[:,indices],np.ravel(train_label))

    
    y_pred=classifier.predict(test_data[:,indices])
    predictions = classifier.predict_proba(test_data[:,indices])

    
    number_rainfall_predicted_landslides=np.sum(y_pred)
    number_earthquake_predicted_landslides=np.shape(y_pred)[0]-number_rainfall_predicted_landslides
    
    probability_earthquake_triggered_inventory=(number_earthquake_predicted_landslides/np.shape(y_pred)[0])*100
    probability_rainfall_triggered_inventory=(number_rainfall_predicted_landslides/np.shape(y_pred)[0])*100
    
    probability_earthquake_triggered_inventory=np.round(probability_earthquake_triggered_inventory,2)
    probability_rainfall_triggered_inventory=np.round(probability_rainfall_triggered_inventory,2)
   
    
    print("Probability of inventory triggered by Earthquake: ", str(float(probability_earthquake_triggered_inventory))+'%')
    print("Probability of inventory triggered by Rainfall: ",str(float(probability_rainfall_triggered_inventory))+'%')
    
    return predictions



def classify_inventory_best(earthquake_inventory_features, rainfall_inventory_features, test_inventory_features):
    """
    Function to classify testing inventory into earthquake and rainfall classes without cross-validation.

    Parameters:
        :earthquake_inventory_features (array_like): TDA features of known earthquake inventories landslides
        :rainfall_inventory_features (array_like): TDA features of known rainfall inventories landslides
        :test_inventory_features (array_like): TDA features of known testing inventory landslides

    Returns:
           (array_like) probability of testing inventory landslides belonging to earthquake and rainfall class
    """

    earthquake_label = np.zeros((np.shape(earthquake_inventory_features)[0], 1))
    rainfall_label = np.ones((np.shape(rainfall_inventory_features)[0], 1))

    n1 = np.shape(earthquake_inventory_features)[0]
    n2 = np.shape(rainfall_inventory_features)[0]

    # Balancing the dataset
    if n1 > n2:  # More earthquake samples
        indi_earth = random.sample(range(n1), n2)
        train_earth = earthquake_inventory_features[indi_earth, :]
        train_label_earth = earthquake_label[indi_earth]
        train_rain = rainfall_inventory_features
        train_label_rain = rainfall_label
    else:  # More rainfall samples
        indi_rain = random.sample(range(n2), n1)
        train_rain = rainfall_inventory_features[indi_rain, :]
        train_label_rain = rainfall_label[indi_rain]
        train_earth = earthquake_inventory_features
        train_label_earth = earthquake_label

    train_data = np.vstack((train_earth, train_rain))
    train_label = np.vstack((train_label_earth, train_label_rain))

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_inventory_features)

    # Define best parameters and selected features
    best_params = {'n_estimators': 777, 'max_depth': 5, 'learning_rate': 0.2086777141244184,
                   'subsample': 0.5785828045682423, 'colsample_bytree': 0.9099546082430142}
    selected_features = [1, 4, 6, 7, 21, 22]

    # Final classifier trained on all training data with selected features
    classifier = XGBClassifier(**best_params, eval_metric='logloss')
    classifier.fit(X_train[:, selected_features], np.ravel(train_label))

    # Predictions for test data
    y_pred = classifier.predict(test_data[:, selected_features])
    predictions = classifier.predict_proba(test_data[:, selected_features])

    # Class probabilities
    number_rainfall_predicted_landslides = np.sum(y_pred)
    number_earthquake_predicted_landslides = np.shape(y_pred)[0] - number_rainfall_predicted_landslides

    probability_earthquake_triggered_inventory = (number_earthquake_predicted_landslides / np.shape(y_pred)[0]) * 100
    probability_rainfall_triggered_inventory = (number_rainfall_predicted_landslides / np.shape(y_pred)[0]) * 100

    probability_earthquake_triggered_inventory = np.round(probability_earthquake_triggered_inventory, 2)
    probability_rainfall_triggered_inventory = np.round(probability_rainfall_triggered_inventory, 2)

    print("Probability of inventory triggered by Earthquake:", str(float(probability_earthquake_triggered_inventory)) + '%')
    print("Probability of inventory triggered by Rainfall:", str(float(probability_rainfall_triggered_inventory)) + '%')

    return predictions

def plot_topological_results(predict_proba):
    
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

    ##################################################################################
    #matrix_probability=RF_image(predict_proba)
    #image=np.dstack((matrix_probability,matrix_probability,matrix_probability))
    #image[matrix_probability[:,:]==0]=[230,204,179]
    #image[matrix_probability[:,:]==1]=[30,100,185]
    #image=np.int32(image)
    ###################################################################################
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



def plot_polygon(poly_data, polygon_index):
    """
    Plot the 2D structure of a polygon from the given data.

    Parameters:
        :poly_data (GeoDataFrame): Polygons shapefile
        :polygon_index (int): Index of the polygon to plot

    Returns:
        None
    """
    # Extract the polygon geometry
    polygon = poly_data['geometry'][polygon_index]
    
    if polygon.geom_type == 'Polygon':
        # Get exterior coordinates of the polygon
        x, y = polygon.exterior.xy
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=f"Polygon {polygon_index}", color="blue")
        plt.fill(x, y, alpha=0.3, color="lightblue")
        plt.title(f"2D Structure of Polygon {polygon_index}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"Geometry at index {polygon_index} is not a Polygon: {polygon.geom_type}")



    ##################################################################################    

