# Required imports
import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Data Processing
def load_data(paths, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print(f'Reading images from {paths}')
    for fld in classes:
        index = classes.index(fld)
        print(f'Loading {fld} files (Index: {index})')
        path_jpg = os.path.join(paths, fld, '*.[jJ][pP][gG]')  # Matches .jpg, .JPG
        path_png = os.path.join(paths, fld, '*.[pP][nN][gG]')  # Matches .png, .PNG
        files = glob.glob(path_jpg)
        files.extend(glob.glob(path_png))
        print(f'Found {len(files)} files in {os.path.join(paths, fld)}')

        if not files:
            print(f"Warning: No image files found in {os.path.join(paths, fld)}")

        for fl in files:
            image = cv2.imread(fl)
            if image is None:
                print(f"Warning: Failed to load image {fl}")
                continue
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)

    if not images:
        raise ValueError(f"No valid images loaded from {paths}. Check directory structure and file formats.")

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls

# Normalising the data
def normalize_data(train_images, validation_images, test_images):
    # Ensure inputs are numpy arrays
    train_images = np.array(train_images, dtype=np.uint8)
    validation_images = np.array(validation_images, dtype=np.uint8)
    test_images = np.array(test_images, dtype=np.uint8)
    
    # Normalize to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    validation_images = validation_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
        
    return train_images, validation_images, test_images

# Splitting into training, testing, validating
def prepare_data(paths, image_size, classes, test_size=0.2, validation_size=0.2, random_state=42):
    # Load data
    images, labels, ids, cls = load_data(paths, image_size, classes)

    # Split the data into training and testing sets
    train_images, test_images, train_labels, test_labels, train_ids, test_ids, train_cls, test_cls = train_test_split(
        images, labels, ids, cls, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Further split the training set into training and validation sets
    train_images, validation_images, train_labels, validation_labels, train_ids, validation_ids, train_cls, validation_cls = train_test_split(
        train_images, train_labels, train_ids, train_cls, test_size=validation_size, random_state=random_state, stratify=train_labels
    )

    return (train_images, train_labels, train_ids, train_cls,
            validation_images, validation_labels, validation_ids, validation_cls,
            test_images, test_labels, test_ids, test_cls)

# Object Oriented Programming
class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]

# Keeping the Datasets
def read_train_sets(train_images, train_labels, train_ids, train_cls, 
                    validation_images, validation_labels, validation_ids, validation_cls):
    class DataSets(object):
        pass

    data_sets = DataSets()
    # Assign DataSet objects to train and valid attributes
    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets

# Making a large function to get the Training & Validation data
def Train_Data_Loader(paths, image_size, classes):
    train_images, train_labels, train_ids, train_cls, validation_images, validation_labels, validation_ids, validation_cls, test_images, test_labels, test_ids, test_cls = prepare_data(paths, image_size, classes)
    train_images, validation_images, test_images = normalize_data(train_images, validation_images, test_images)
    data_sets = read_train_sets(train_images, train_labels, train_ids, train_cls, validation_images, validation_labels, validation_ids, validation_cls)
    
    return data_sets

# Making a large function to get the Testing data
def Test_Data_Loader(paths, image_size, classes):
    train_images, train_labels, train_ids, train_cls, validation_images, validation_labels, validation_ids, validation_cls, test_images, test_labels, test_ids, test_cls = prepare_data(paths, image_size, classes)
    train_images, validation_images, test_images = normalize_data(train_images, validation_images, test_images)
    
    return test_images, test_ids