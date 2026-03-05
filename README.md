# face-recognition-pca-filter-wrapper-knn
Implementation of a face recognition system using the ORL Face Dataset. The project compares PCA-based feature extraction, ANOVA filter feature selection, and wrapper methods with k-NN classification using Python and Scikit-learn.
# Face Recognition using PCA, Filter and Wrapper Methods with k-NN

## Overview

This project implements a basic **face recognition system** using the **ORL Face Dataset**. The objective is to study and compare different **dimensionality reduction techniques** before classification using the **k-Nearest Neighbour (k-NN)** algorithm.

The experiment compares three approaches:

1. **PCA (Principal Component Analysis)** – Feature Extraction
2. **Filter Method (ANOVA – SelectKBest)** – Feature Selection
3. **Wrapper Methods (Forward & Backward Selection)** – Feature Selection

The performance of each approach is evaluated using **classification accuracy**.

---

## Dataset

The experiment uses the **ORL Face Dataset**.

**Dataset Characteristics**

* 40 subjects
* 10 images per subject
* Total images: 400
* Image resolution: **92 × 112 pixels**
* Each image converted to grayscale
* Each image flattened into **10304 pixel features**

Feature matrix:

```
X shape = (400, 10304)
y shape = (400,)
```

Where:

* **X** → Pixel intensity features
* **y** → Class labels (person IDs)

---

## Project Workflow

The system follows the steps below:

1. Mount Google Drive in Google Colab
2. Load ORL dataset
3. Convert images to grayscale and flatten
4. Normalize pixel values
5. Split dataset into training and testing sets
6. Apply dimensionality reduction methods
7. Train **k-NN classifier**
8. Evaluate recognition accuracy
9. Compare results

---

## Methods Implemented

### 1. PCA (Feature Extraction)

Principal Component Analysis reduces dimensionality by projecting the data onto directions of maximum variance.

Reduction:

```
10304 → 100 features
```

These new features are known as **Eigenfaces**.

---

### 2. Filter Method – ANOVA (SelectKBest)

Filter methods select the most relevant features based on statistical tests.

The **ANOVA F-test** measures how strongly each feature separates the classes.

Reduction:

```
10304 → 100 selected features
```

---

### 3. Wrapper Methods

Wrapper methods evaluate subsets of features using classifier performance.

Two approaches were used:

**Forward Selection**

* Starts with zero features
* Adds features iteratively

**Backward Selection**

* Starts with all features
* Removes least useful features

To reduce computational cost:

```
10304 → 100 (initial filter)
100 → 50 (wrapper selection)
```

---

## Classifier Used

### k-Nearest Neighbour (k-NN)

* k = 3
* Distance metric: **Euclidean distance**

Classification is performed by finding the nearest neighbours in feature space.

---

## Experimental Results

| Method                 | Accuracy   |
| ---------------------- | ---------- |
| PCA + kNN              | **94.17%** |
| SelectKBest + kNN      | 58.33%     |
| Forward Wrapper + kNN  | 60.83%     |
| Backward Wrapper + kNN | 55.00%     |

---

## Observations

* **PCA achieved the highest accuracy** because it captures global facial structure through eigenfaces.
* Filter methods treat pixels independently, which may not capture spatial relationships in images.
* Wrapper methods are computationally expensive and may not perform well with raw pixel features.

Overall, **feature extraction techniques like PCA are better suited for high-dimensional image data**.

---

## Technologies Used

* Python
* Google Colab
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

---

## Folder Structure

Example dataset structure:

```
ORL/
   s1/
      1.pgm
      2.pgm
   s2/
      1.pgm
      2.pgm
   ...
   s40/
```

---

## How to Run

1. Upload the ORL dataset to **Google Drive**
2. Open the notebook in **Google Colab**
3. Mount Google Drive
4. Set the dataset path
5. Run the cells sequentially

---

## Future Improvements

The system can be improved by:

* Using **Linear Discriminant Analysis (LDA)**
* Using **Support Vector Machines (SVM)**
* Applying **Deep Learning models (CNN)**
* Using **data augmentation techniques**

---

## References

1. ORL Face Database – AT&T Laboratories Cambridge
2. Scikit-learn Documentation
3. Bishop, C. M. *Pattern Recognition and Machine Learning*

---

## Author

Pattern Recognition Assignment – Face Recognition using PCA and Feature Selection
