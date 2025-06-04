# Water Quality Prediction Project using Deep Learning Neural Networks

## Overview
This project uses Deep Learning Neural Network models to predict water quality based on data from the Central Pollution Control Board (CPCB) of India. The dataset contains water quality monitoring data from across India, with chemical and physical parameters measured at various locations over the years 2019-2022.

## Dataset
The dataset includes water quality monitoring data with the following characteristics:
- **Source**: Central Pollution Control Board (CPCB) of India
- **Time Period**: 2019-2022
- **Coverage**: Various locations across India
- **Parameters**: Chemical and physical water quality indicators

## Models
This project implements various Deep Learning Neural Network models to predict water quality parameters:
- **Regression Model**: Predicts a single target quality metric
- **Multi-output Regression Model**: Predicts multiple water quality metrics simultaneously
- **Classification Model**: Classifies water as potable or non-potable

The following trained models are included:
- `best_regression_model.h5`
- `best_multi_output_model.h5`
- `best_classification_model.h5`

## Project Structure
```bash
.
├── Water_Quality_Prediction_Project_using_Deep_Learning_Neural_Networks.ipynb
├── best_regression_model.h5
├── best_multi_output_model.h5
├── best_classification_model.h5
├── README.md
```

## Technologies
- Python
- TensorFlow/Keras
- Pandas
- Numpy
- Scikit-learn
- Matplotlib/Seaborn for visualization

## Installation
```bash
# Clone the repository
git clone https://github.com/manola1109/Water-Quality-Prediction-Project-using-Deep-Learning-Neural-Networks.git

# Navigate to the project directory
cd Water-Quality-Prediction-Project-using-Deep-Learning-Neural-Networks

# Install required packages
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook `Water_Quality_Prediction_Project_using_Deep_Learning_Neural_Networks.ipynb`
2. Follow the workflow from data preprocessing to model training and evaluation
3. Use the trained models for water quality prediction on new data

## Results

### Model Performance Metrics

#### 1. Regression Model (For Water Quality Index prediction)
- Mean Squared Error: 378,327,232.8783
- Root Mean Squared Error: 19,450.6358
- R² Score: -3564.1409
- Mean Absolute Error: 434.0099

#### 2. Classification Model (For Water Quality Classification)
- Accuracy: 0.9229
- F1 Score: 0.9096
- Precision: 0.9203
- Recall: 0.9229

#### 3. Multi-output Model
- Regression R² Score: -29.9920
- Classification Accuracy: 0.8112

#### 4. Model Comparison
- Multi-output model performed better for WQI prediction
- Single-task model performed better for water quality classification

### Key Features for Water Quality
Feature importance analysis shows these are the most important factors:

Top 5 features for WQI prediction:
1. EC
2. TDS
3. Latitude
4. TDS_EC_ratio
5. Longitude

### Visualizations

#### Confusion Matrix
```
Confusion Matrix:
Rows: Actual Class, Columns: Predicted Class
==================================================
Excellent             2     112       0       0       0
Good                  1     229      15       0       0
Poor                  1      27     768       0       2
Unsuitable for        0       0       0     987       4
Very Poor yet D       0       0      33      25     649
```

Accuracy per class:
- Excellent: 0.0175
- Good: 0.9347
- Poor: 0.9624
- Unsuitable for: 0.9960
- Very Poor yet D: 0.9180

![Model Performance Comparison](images/model_comparison.png)
*Figure 1: Comparison of different model performances*

![Actual vs. Predicted Values](images/actual_vs_predicted.png)
*Figure 2: Scatter plot of actual versus predicted water quality values*

![Feature Importance](images/feature_importance.png)
*Figure 3: Importance of different features in predicting water quality*

### Recommendations
- Deploy the multi-output model for best overall performance
- Focus monitoring efforts on the most predictive parameters
- Consider regional factors when interpreting water quality predictions

## Future Work
- Incorporate spatial analysis to account for geographical variations
- Implement ensemble methods to improve prediction accuracy
- Extend the model to predict water quality in real-time using streaming data
- Integrate with IoT devices for continuous water quality monitoring

## License
This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2025 Deepak Singh Manola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact
For questions or collaboration opportunities, please contact:
- **GitHub**: [@manola1109](https://github.com/manola1109)

## Acknowledgments
- Central Pollution Control Board (CPCB) of India for providing the dataset
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) teams for their excellent deep learning frameworks
- The open-source community for their invaluable tools and libraries

## Project Status
Last updated: 2025-06-04 14:05:55 UTC by manola1109
