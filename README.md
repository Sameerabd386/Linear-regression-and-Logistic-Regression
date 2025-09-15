# Linear-regression-and-Logistic-Regression
A practical guide to linear and logistic regression.


# Linear Regression

Linear regression  is a supervised machine learning algorithm used for predicting continuous values. It works by finding the best-fit straight line (or a hyperplane in higher dimensions) that represents the relationship between a dependent variable and one or more independent variables. The goal is to minimize the distance between the data points and this line, allowing for accurate predictions. A common use case is predicting house prices based on features like square footage and number of bedrooms.

# Logistic Regression

Logistic regression, despite its name, is used for predicting categorical or discrete values, making it a classification algorithm. It works by using a logistic function to model a binary outcome, like "yes" or "no," "true" or "false," or "spam" or "not spam." Instead of predicting a continuous number, it outputs a probability score that the given input belongs to a certain class. This probability is then used to classify the input.

# California Housing Prices: A Linear Regression Model

This notebook focuses on predicting median house values in California using the well-known California Housing Dataset from scikit-learn. It demonstrates a complete linear regression workflow, from initial data handling to model evaluation and making a final prediction on new data.

Here's a breakdown of what the code does:

**Data Loading & Exploration:** The script begins by loading the fetch_california_housing dataset into a pandas DataFrame. It then performs initial exploration, including checking the shape, data types, and descriptive statistics to get a feel for the data.

**Data Splitting:** The dataset is split into training and testing sets. This is a crucial step in machine learning to ensure the model is evaluated on data it has not seen before, providing a more reliable measure of its performance. The MEDV column (Median House Value) is designated as the target variable.

**Model Training:** A LinearRegression model is instantiated and trained on the training data. The model learns the linear relationship between the input features (like median income, house age, etc.) and the target house price.

**Model Evaluation:** The trained model is used to predict house prices on the test set. The performance is then evaluated using the Mean Squared Error (MSE), a common metric for regression tasks. The code also prints the model's coefficients and intercept, which represent the weights and bias of the linear equation.

**Visualization:** A scatter plot is generated to visually compare the model's predicted values against the actual house prices. A diagonal line is added to the plot, representing the ideal scenario of perfect prediction. This visualization helps in understanding how well the model's predictions align with the real-world data.

Prediction on New Data: Finally, the code shows how to use the trained model to make a prediction on a new, hypothetical data point. This demonstrates the practical application of the model in a real-world scenario.


# Handwritten Digit Classification: A Logistic Regression Model

This notebook demonstrates the use of Logistic Regression to classify handwritten digits (0–9) using the classic Digits Dataset from scikit-learn. It walks through the entire machine learning workflow, from data preparation to model evaluation and prediction on new samples.

Here’s a breakdown of what the code does:

**Data Loading & Exploration:**
The script loads the load_digits dataset into a pandas DataFrame. The dataset contains 1,797 images of handwritten digits, each represented as an 8x8 pixel grid (64 features in total). A target column is added to store the actual digit labels. Initial exploration (like .head() and .shape) helps in understanding the dataset structure.

**Data Splitting:**
The dataset is split into training and testing sets using train_test_split, with 20% of the data reserved for testing. Stratified sampling ensures that all digit classes are proportionally represented in both train and test sets.

**Model Training:**
A LogisticRegression model is instantiated with max_iter=1000 to ensure convergence. The model is trained on the training set, where it learns to distinguish between different digits based on pixel intensity values.

**Model Evaluation:**
The trained model makes predictions on the test set. Accuracy is used as the evaluation metric, which measures the percentage of correctly classified digits. The final accuracy score provides insight into how well the model generalizes to unseen data.

**Prediction on New Data:**
The code demonstrates how to make a prediction on a manually modified data point. A single digit image (from the dataset) is altered slightly, reshaped, and passed through the model for classification. This shows how the trained model can be applied in practice to predict digits from raw pixel data.
