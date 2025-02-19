import numpy as np
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
import time
#import keras 
#import tensorflow
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


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
    function to convert the (longitude latitude) coordinates of polygons to UTM(easting, northing) coordinates
    
    
    Parameters:
          :lonllat_polydata (array_like): 
                             longitude and latitude coordinates data
                      
    Returns:
            (array_like)
            easting and northing coordinates of landslide polygon data when polygon data has longitude latitude coordinates 

    i.e., converts geographic coordinates to Cartesian coordinates).
     
    """
     
    east_north_polydata=[]  # initializes an empty list to store the converted coordinates.
    
    #np.shape(lonlat_polydata)[0]: Gets the number of coordinate pairs in the data. 
    # For example, if lonlat_polydata is a list of 10 points, np.shape(lonlat_polydata)[0] will return 10

    for i in range(np.shape(lonlat_polydata)[0]):  # A for loop is used to iterate over each point in lonlat_polydata.
        u = utm.from_latlon(lonlat_polydata[i][1], lonlat_polydata[i][0])   ### (lat,lon) to (east,north)
        east_north_polydata.append([u[0],u[1]]) #Appends the resulting (easting, northing) coordinate to the east_north_polydata list.
    east_north_polydata=np.asarray(east_north_polydata) #Converts east_north_polydata to a NumPy array and returns it
    return  east_north_polydata  

def increase_resolution_polygon(data):
    """
    function to increase the data points between two neighbouring vertex of landslide polygon to get smooth images
    
    Parameters:
            :data (array_like):
                               easting and northing coordinates data of landslide polygon
     
    Returns:
            (array_like)
            linear interpolated data of landslide polygon
    """
    N=100 #N is the number of points you want between two neighboring vertices. This increases the resolution by adding more points between each original pair.
    n=np.shape(data)[0]-1  #n is the number of edges between the vertices (since the shape of data gives the number of vertices).
    dat=[]   # dat initializes an empty list to store interpolated points.
    for j in range(n):
        x1,y1=data[j]  # coordintaes of starting points of an edge
        x2,y2=data[j+1] # coordinates of ending points of an edge
        x_dis,y_dis=np.abs(x1-x2),np.abs(y1-y2)      #    x_dis and y_dis are the absolute differences between the coordinates of the two points.
        x=min(x1,x2) # x is the minimum of x1 and x2. It is used to ensure we interpolate between the minimum and maximum coordinates.
        for i in range(1,N):    # This nested loop adds interpolated points between x1, y1 and x2, y2.
            xx=x+(x_dis/N)*i 
            if x2 != x1:
                 yy = y1 + ((y2 - y1) / (x2 - x1)) * (xx - x1)  # yy is calculated using linear interpolation.
            else:
             yy = y1  # Set yy to y1 when x1 == x2 (i.e., a vertical line)
   
            #yy=y1+((y2-y1)/(x2-x1))*(xx-x1)    # yy is calculated using linear interpolation.
            dat.append([xx,yy])    #then append the coordinates
            
    dat=np.asarray(dat)    # Converts dat to a NumPy array and returns it.
    return dat

def make_ls_images(poly_data):
    
    """
    function to convert landslide polygon to images
    
    Parameters:
          :poly_data :readed landslide inventory shapefile 
                
    Returns:
        (array_like) Bnary pixels values of landslide polygon Image 
    """   
    
    DATA=[]  #Initializes an empty list DATA to store the images.  
    
    for l in range(np.shape(poly_data)[0]):   # Iterates over each row of poly_data.

        if poly_data['geometry'][l].geom_type=='Polygon':   # Checks if the geometry type is 'Polygon'.

            polygon_data=np.asarray(poly_data['geometry'][l].exterior.coords)  # Converts the exterior coordinates of the polygon to a NumPy array.
             
            if np.nanmin(polygon_data) < 100: # If the minimum value in polygon_data is less than 100, it is assumed to be in latitude/longitude 
                #(since latitudes/longitudes are typically below 100 degrees).
                # Converts polygon_data to easting/northing using the latlon_to_eastnorth function.
               polygon_data=latlon_to_eastnorth(polygon_data)   
            
            polygon_data=polygon_data[~np.isnan(polygon_data).any(axis=1)]      ### remove any Nan Values in polygon
            ####################################################################################################
            polygon_new_data=increase_resolution_polygon(polygon_data)   # Uses increase_resolution_polygon to add more points to polygon_data.
            x_new=polygon_new_data[:,0]-np.nanmin(polygon_new_data[:,0])  # Subtracts the minimum values from x coordinates to normalize them to start from zero
            y_new=polygon_new_data[:,1]-np.nanmin(polygon_new_data[:,1])  # Subtracts the minimum values from y coordinates to normalize them to start from zero

            ########### CHECK THE CODE ################################
            if (np.max(x_new)<180) & (np.max(y_new)<180):  #Checks if the maximum values of x_new and y_new are less than 180 (ensuring the polygon fits within the image size).

                div=3
                #x1,y1=np.int32(x_new/div),np.int32(y_new/div)
                x1,y1=np.around(x_new/div),np.around(y_new/div) #Divides x_new and y_new by 3 to scale down the size and then rounds and converts them to integers.
                x1,y1=np.int32(x1),np.int32(y1)  
                
                k1,k2=32-int(np.max(x1)/2),32-int(np.max(y1)/2)  #    k1 and k2 are calculated to center the polygon in a 64x64 grid.
                x1,y1=x1+k1,y1+k2   # Adds k1 and k2 to x1 and y1 to shift the polygon to the center.
                
                
            #    Creates a 64x64 black image (image).
            #  Sets the corresponding pixels of the polygon to 255 (white).
            #  Flips the image vertically to correct the orientation.
            #  Appends the image to DATA.
                image=np.zeros((64,64))
                for i,j in zip(x1,y1):  
                    image[j,i]=255
                image=np.flip(image,axis=0)    
                DATA.append(image)
    DATA=np.asarray(DATA) #Converts DATA to a NumPy array and returns it.
    print(np.shape(DATA))
    
    """
    output of the function is binaryscale (64 x 64) images
    
    """
    return DATA   


def visualize_images(data, num_images=5, cmap='gray', figsize=(15, 5)):
    """
    Visualize a set of images from the given dataset.

    Parameters:
        data (list or array): The dataset containing the images to be visualized.
        num_images (int): The number of images to display.
        cmap (str): Colormap for the images (default is 'gray').
        figsize (tuple): Figure size for the plot.

    Returns:
        None
    """
    # Create a plot to visualize the images
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    for i in range(num_images):
        # Ensure the index is within the range of data
        if i < len(data):
            axes[i].imshow(data[i], cmap=cmap)
            axes[i].axis('off')
            axes[i].set_title(f'Image {i+1}')
        else:
            # Hide axes if no more images
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# visualize_images is a function for the visual representation of the image of the landslide polygons



def train_augment(train_data,train_label):
    """
    
    This function is used to augment the training data by rotating image by 90, 180, 270 degree and flippong image horizontally and
    vertically
    
    Parameters:
              :train_data (array_like): training data
              :train_label (array_like): training label
    
    Returns:
           augmented training data and labels
    
    
    """
    
    
    new_train=[]  #Initializes an empty list to store augmented training images.
    new_train_label=[]  # Initializes an empty list to store the corresponding labels for augmented images.
    for i in range(np.shape(train_data)[0]): #Loops over each training image.
        aa=train_data[i,:,:] # Extracts the i-th image from train_data. This is a 2D array representing pixel values of an image.
        bb=train_label[i]    # Extracts the corresponding label for the i-th image.
        new_train_label.append(bb) # Appends the label bb to new_train_label.
        new_train.append(aa)       # Appends the original image aa to new_train.
        
        
        aa_1=np.fliplr(aa)          # Flips the image aa horizontally (left to right).
        new_train.append(aa_1)      # Appends the horizontally flipped image to new_train
        new_train_label.append(bb)  # Appends the label bb to new_train_label. 
        # The label remains the same since the transformation doesn't change the class of the image.


        aa_2=np.flipud(aa)         #  Flips the image aa vertically (upside down).
        new_train.append(aa_2)     # Appends the vertically flipped image to new_train.
        new_train_label.append(bb)  #Appends the label bb to new_train_label
        
        bb_1=np.rot90(aa)           # Rotates the image aa by 90 degrees counterclockwise.
        new_train.append(bb_1)
        new_train_label.append(bb)  

        cc_1=np.rot90(bb_1)       # Rotates the previously rotated image (bb_1) by 90 degrees again, resulting in a total rotation of 180 degrees.
        new_train.append(cc_1)
        new_train_label.append(bb)

        dd_1=np.rot90(cc_1)        #Rotates the previously rotated image (cc_1) by 90 degrees again, resulting in a total rotation of 270 degrees.
        new_train.append(dd_1) 
        new_train_label.append(bb)     
        
    new_train=np.asarray(new_train)[:,:,:]  #Finally, converting the lists to NumPy arrays ensures the output is compatible with machine learning models, which typically require NumPy arrays as input.
    new_train_label=np.asarray(new_train_label)[:,:]  

    return new_train,new_train_label


'''def classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features):
    
    """
    function to give probability of testing inventory belonging to earthquake and rainfall class
    
    Parameters:
        :earthquake_inventory_features (array_like): is landslide images of known earthquake inventories landslides
        :rainfall_inventory_features (array_like): is landslide images of known rainfall inventories landslides
        :test_inventory_features (array_like): is landslide images of known testing inventory landslides  
                
    Returns:
            (array_like) probability of testing inventory landslides belonging to earthquake and rainfall class

        
    """
    
     # Create labels for the two classes. Labels for earthquake samples are set to 0, and for rainfall samples, they are set to 1.
    earthquake_label=np.zeros((np.shape(earthquake_inventory_features)[0],1))
    rainfall_label=np.ones((np.shape(rainfall_inventory_features)[0],1))
   

    n1=np.shape(earthquake_inventory_features)[0]
    n2=np.shape(rainfall_inventory_features)[0]
    
    # Random Sampling for Balanced Training Set
    # If there are more earthquake samples than rainfall samples, it randomly samples from the earthquake dataset to match the number of rainfall samples.

    if n1>n2:  ### n1 is number of earth samples and n2 is number of rainfall samples #####
        indi_earth=random.sample(range(n1),n2)  #  randomly selects indices from the earthquake dataset.
        train_earth=earthquake_inventory_features[indi_earth,:]
        train_label_earth=earthquake_label[indi_earth]  

        train_rain=rainfall_inventory_features
        train_label_rain=rainfall_label
        #######################################################################################
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

    else: # If there are more rainfall samples than earthquake samples, it randomly samples from the rainfall dataset to match the number of earthquake samples.
        indi_rain=random.sample(range(n2),n1)
        train_rain=rainfall_inventory_features[indi_rain,:]
        train_label_rain=rainfall_label[indi_rain]   
        train_earth=earthquake_inventory_features
        train_label_earth=earthquake_label        
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

        #########################################################################################
    
    # The train_augment function is called to perform data augmentation on the training set,
    #  which helps improve the model's robustness by artificially increasing the size and diversity of the training data
    x_train,y_train=train_augment(train_data,train_label)

    # x_test is assigned the testing inventory features
    x_test=test_inventory_features
    #y_test=test_inventory_labels

    #img_rows and img_cols specify the input image dimensions.
    # dim specifies the number of channels (1 for grayscale).
    img_rows, img_cols,dim = 64, 64,1

  # hyperparameters
    batch_size=64
    epochs=30
    num_classes=2
    input_shape=(64,64,1)
    
    num_classes=2
    img_rows, img_cols,dim = 64, 64,1
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    # converts labels into a one-hot encoded format suitable for categorical classification.
    
    #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype('float32')  # this is a common format for image data in neural networks.
    x_test = x_test.astype('float32') 
    #x_train = np.expand_dims(x_train, axis=-1)
    #x_test = np.expand_dims(x_test, -1)
    
    
    model = Sequential()
    # Sequential() creates a linear stack of layers for the model.
    #model.add(Input(shape=(64, 64, 1))) 
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64,1)))
    # Conv2D adds convolutional layers with 32 filters of size 3×33×3 and the ReLU activation function.
    
    model.add(MaxPooling2D((2, 2)))
    #  reduces the spatial dimensions of the feature maps, helping to control overfitting and reduce computation.

    layers.Dropout(0.25),

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
   # converts the 2D feature maps to 1D, preparing the data for the dense layers.

    layers.Dropout(0.25),

    model.add(Dense(40, activation='relu'))
    #  adds a fully connected layer with 40 units.

    layers.Dropout(0.25),
    model.add(Dense(2, activation='softmax'))
    #  layer outputs probabilities for the two classes (earthquake and rainfall) using the softmax function.
    #model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    hist=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0) 
    
    predictions=model.predict(x_test)
    
    """
       This functions print the probability of inventory belonging to earthquake and rainfall class
       
       Output of function is probability of each landslide in inventory class belonging to earthuake and rainfall class.
       The output of function is used for visualization of testing results
    
    """
    
    
    return predictions

'''



import numpy as np
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Define hyperparameters
best_hyperparams = {
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.0006736171423539479,
    "num_filters": 32,
    "dense_units": 40,
    "dropout_rate": 0.3403095137594212
}



def classify_inventory_cnn(earthquake_inventory_features, rainfall_inventory_features, test_inventory_features):
    """
    Classifies landslides into earthquake or rainfall triggered using CNN.
    
    Parameters:
        earthquake_inventory_features (array_like): Images of known earthquake-triggered landslides.
        rainfall_inventory_features (array_like): Images of known rainfall-triggered landslides.
        test_inventory_features (array_like): Images of unknown test landslides.
    
    Returns:
        array_like: Probability of test landslides belonging to earthquake or rainfall class.
    """

    # Create labels for earthquake (0) and rainfall (1)
    earthquake_label = np.zeros((earthquake_inventory_features.shape[0], 1))
    rainfall_label = np.ones((rainfall_inventory_features.shape[0], 1))

    # Get counts of samples
    n1 = earthquake_inventory_features.shape[0]  # Number of earthquake samples
    n2 = rainfall_inventory_features.shape[0]  # Number of rainfall samples

    # Balance dataset by randomly sampling from larger class
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

    # Combine balanced training data
    train_data = np.vstack((train_earth, train_rain))
    train_label = np.vstack((train_label_earth, train_label_rain))

    # Data augmentation
    x_train, y_train = train_augment(train_data, train_label)
    
    # Reshape inputs for CNN
    img_rows, img_cols, dim = 64, 64, 1
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = test_inventory_features.reshape(test_inventory_features.shape[0], img_rows, img_cols, 1)

    # Convert labels to categorical format
    num_classes = 2
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    # Normalize input data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define CNN model
    model = Sequential([
        Conv2D(best_hyperparams["num_filters"], (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Dropout(best_hyperparams["dropout_rate"]),

        Conv2D(best_hyperparams["num_filters"], (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(best_hyperparams["dropout_rate"]),

        Flatten(),
        Dense(best_hyperparams["dense_units"], activation='relu'),
        Dropout(best_hyperparams["dropout_rate"]),
        
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=best_hyperparams["learning_rate"]),
        metrics=["accuracy"]
    )

    # Train model
    model.fit(
        x_train, y_train,
        batch_size=best_hyperparams["batch_size"],
        epochs=best_hyperparams["epochs"],
        verbose=1
    )

    # Make predictions
    predictions = model.predict(x_test)

    return predictions

 
###################################################################################################################################    
def plot_geometric_results(predict_proba):
    
    """
    function to visualize the trigger prediction of landslides in testing inventory
    
     
    Parameters:
         :predict_proba (array_like): probability of each landslide in inventory class belonging to earthquake and rainfall class.
                   
                   
    Returns:
         Visualization of landslide probabilities belong to earthquake and rainfall class and trigger prediction of entire landslide 
         inventory 
                
    """
    
    plt.rc('text', usetex=False)
    # chage xtick and ytick fontsize 
    # chage xtick and ytick fontsize 
    #plt.rc('text', usetex=True)
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['font.family'] = 'sans-serif'

    plt.rcParams['pdf.fonttype'] = 42

    #n1 and n2: Count how many probabilities are greater than 0.5 for each of the two classes (earthquake and rainfall).
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
    cb.ax.set_xticklabels([],size=0)                 # vertically oriented colorbar
    cb.ax.set_yticklabels([],size=0)                 # vertically oriented colorbar
    cb.ax.tick_params(axis=u'both', which=u'both',size=0)
    ax.text(np.shape(predict_proba)[0]//6,135,'Earthquake',fontsize=26)
    ax.text(np.shape(predict_proba)[0]//1.4,135,'Rainfall',fontsize=26)

    #cb.set_label('Earthquake                            Rainfall ',fontsize=26)
    plt.show()
    ##################################################################################    

    ##################################################################################    







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




 


