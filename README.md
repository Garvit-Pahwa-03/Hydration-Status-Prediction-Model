# Hydration Status Prediction Model

This repository contains a prototype for predicting hydration status, developed as part of an AI/ML Engineer Intern selection process for Runverve. The project explores two different supervised machine learning models: XGBoost and Random Forest, both tuned using Optuna.

## Project Overview

The goal of this project is to build and compare supervised machine learning models that can predict an individual's hydration level (Well Hydrated, Mildly Dehydrated, Dehydrated) based on simulated physiological and environmental factors.

## Methodology

### 1. Dataset
Due to the difficulty in finding readily available public datasets with explicit hydration labels for this rapid prototyping task, this project utilizes **synthetically generated data**.

*   **Features Simulated:**
    *   `HeartRate (bpm)`: Heart rate, which tends to increase with dehydration.
    *   `BodyTemperature (°C)`: Core body temperature, which can increase.
    *   `AmbientTemperature (°C)`: Environmental temperature, a factor in sweat loss.
    *   `ActivityLevel (0-10 scale)`: Intensity of physical activity, influencing fluid loss.
    *   `FluidIntake_LastHour (mL)`: Amount of fluid consumed in the past hour.
    *   `TimeSinceLastDrink (hours)`: Duration since the last fluid intake.
*   **Target Variable:**
    *   `HydrationStatus`: A categorical variable with three levels:
        *   0: Well Hydrated
        *   1: Mildly Dehydrated
        *   2: Dehydrated
*   **Generation Logic:** The data generation script (within `Runverve_Task_R2.ipynb`) simulates plausible physiological responses. For example, high activity levels, high ambient temperatures, and prolonged time since the last drink contribute to a higher "dehydration score," which then influences heart rate, body temperature, and the final hydration status label. Randomness is introduced to create variability.

### 2. Data Preprocessing
The following preprocessing steps were applied:
*   **Train-Test Split:** The dataset was split into training (75%) and testing (25%) sets using `stratify=Y` to maintain class proportions in both sets.
*   **Feature Scaling:** `StandardScaler` from Scikit-learn was used to standardize the numerical features (mean 0, standard deviation 1). This is generally good practice and can help some algorithms perform better.

### 3. Feature Selection Insights
*   **Correlation Analysis:** A heatmap was generated to visualize the correlation between features and the target variable on the training data. This confirmed that the engineered features had the expected relationships with hydration status (e.g., `FluidIntake_LastHour` negatively correlated, `TimeSinceLastDrink` positively correlated).
*   **Model-Based Importance:** After training each tuned model (XGBoost and Random Forest), its `feature_importances_` attribute was used to identify the most influential features. For this prototype, all generated features were retained for model training.

### 4. Models and Hyperparameter Tuning

Two supervised classification models were implemented and tuned:

**a. XGBoost (Extreme Gradient Boosting) Classifier**
*   **Reasoning:** Chosen for its high performance on structured/tabular data, efficiency, and built-in feature importance.
*   **Configuration:** `objective='multi:softmax'`, `num_class=3`.
*   **Hyperparameter Tuning:** Optuna was used to optimize hyperparameters such as `booster`, `lambda`, `alpha`, `subsample`, `colsample_bytree`, `max_depth`, `min_child_weight`, `eta`, `gamma`, and `n_estimators`. The objective function for Optuna used 3-fold `StratifiedKFold` cross-validation on the training set to evaluate hyperparameter combinations, aiming to maximize mean CV accuracy. An initial model with default parameters was also trained for baseline comparison.

**b. Random Forest Classifier**
*   **Reasoning:** A robust ensemble method, good for classification tasks, and also provides feature importances.
*   **Hyperparameter Tuning:** Optuna was used to find optimal values for `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`. Similar to XGBoost, 3-fold `StratifiedKFold` cross-validation was employed within the Optuna objective function.

### 5. Evaluation
Each tuned model's performance (and the baseline XGBoost model) was evaluated on the held-out test set using:
*   **Accuracy:** Overall correctness of predictions.
*   **Classification Report:** Provided precision, recall, and F1-score for each hydration class ('Well Hydrated', 'Mildly Dehydrated', 'Dehydrated').
*   **Confusion Matrix:** Visualized true vs. predicted labels to understand misclassifications.

**Performance Note:** The models generally achieve high performance (accuracy > 90% for tuned models). This is characteristic of models trained on synthetic data where the underlying patterns are well-defined by the generation rules. Real-world data would present more complex challenges.

## How to Run

1.  **Environment:**
    *   This notebook is designed to run in Google Colab or a local Jupyter environment.
    *   Python 3.7+ is recommended.
2.  **Prerequisites & Installation:**
    *   The first cell in the notebook installs `optuna`. The `tensorflow` installation in that cell is not strictly required for the models implemented but was included.
        ```python
        !pip install optuna
        # !pip install tensorflow # (TensorFlow not used for current models)
        ```
    *   Other required libraries are standard in data science: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`. These are typically pre-installed in Colab or can be installed via pip:
        ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn xgboost
        ```
3.  **Execution:**
    *   Download or clone the `Runverve_Task_R2.ipynb` notebook file.
    *   Open it in Google Colab (upload) or a local JupyterLab/Jupyter Notebook session.
    *   Run the cells sequentially from top to bottom.
    *   **Note:** The Optuna hyperparameter tuning cells can take some time to execute depending on the number of trials (50 trials were set for XGBoost, 50 for Random Forest in the notebook).

## Integration Idea with Digital Twin

A Digital Twin (DT) is a dynamic virtual representation of a physical asset, process, or system. For human health, a DT could model an individual's physiological state.

**Integrating Hydration Prediction with a Digital Twin:**

1.  **Data Ingestion for DT:**
    *   The DT would receive continuous or periodic data from the individual's wearables (smartwatch, fitness bands, skin patches):
        *   Real-time: Heart Rate, HRV, Skin Temperature (proxy for core body temp), Activity (accelerometer/gyroscope data).
        *   Environmental: Ambient temperature (from wearable or phone).
    *   Manual inputs via a companion app:
        *   Fluid intake (volume, type, timestamp).
        *   Urine color/frequency (subjective hydration proxies).
        *   Subjective feelings: thirst, fatigue levels.

2.  **Hydration Models as DT Modules:**
    *   The trained XGBoost or Random Forest model (refined with real-world, personalized data) would be an intelligent module within the DT.
    *   The DT would feed the ingested data (preprocessed appropriately) to the selected model to get a real-time or near real-time hydration status prediction.

3.  **DT Functionalities & Benefits with Hydration Module:**
    *   **Personalized Alerts & Recommendations:**
        *   If the DT's hydration module predicts a decline in hydration (e.g., moving to "Mildly Dehydrated"), it can trigger personalized alerts on the user's device: "Predicted hydration dropping. Activity level high. Consider drinking 300ml water in the next 30 mins."
        *   Recommendations can be dynamically adjusted based on current activity, upcoming planned activities (from calendar), ambient conditions, and historical hydration patterns stored in the DT.
    *   **"What-If" Scenarios & Proactive Planning:**
        *   Users could query the DT: "I'm planning a 10km run tomorrow at 2 PM, expected temp 30°C. What's my hydration strategy?" The DT, using the hydration model and potentially other physiological models (e.g., for sweat rate), could simulate fluid loss and recommend pre-hydration, intra-activity intake, etc.
    *   **Performance & Safety Optimization:**
        *   For athletes or workers in physically demanding environments (e.g., at Runverve), the DT can monitor hydration to optimize performance and prevent dehydration-related issues like heat stress or cognitive decline.
    *   **Longitudinal Health Tracking & Insights:**
        *   The DT would store historical hydration predictions and input data, allowing for trend analysis. It could identify patterns like, "You often become mildly dehydrated by 3 PM on workdays when you don't have your scheduled water break."
    *   **Adaptive Model Personalization:**
        *   The DT can facilitate continuous learning. User feedback (e.g., confirming/denying feeling dehydrated) or occasional objective measures (e.g., urine specific gravity test) can be used to retrain and fine-tune the embedded hydration model for improved individual accuracy over time.
    *   **Holistic Well-being Correlation:**
        *   The DT could correlate predicted hydration status with other tracked metrics like sleep quality, stress levels, or predicted fatigue. For example, it might discover that even mild dehydration significantly impacts the user's perceived fatigue or next-day recovery.

**Technical Implementation Sketch:**
*   **Data Layer:** Wearable sensors stream data to a secure cloud platform via a mobile app.
*   **Digital Twin Core:** A cloud-hosted service representing the individual's DT instance. It manages data, state, and models.
*   **Prediction Service:** The trained ML models (XGBoost or RF) deployed as API endpoints (e.g., using Flask/FastAPI, or serverless functions). The DT Core calls these services with preprocessed data.
*   **User Interface:** Mobile app for manual input, displaying insights, alerts, and recommendations from the DT.

This integration would elevate the predictive models from standalone tools to an integral part of a proactive, personalized health and performance management system within the Runverve ecosystem.

## Future Work / Potential Improvements for the Prototype
*   **More Extensive Hyperparameter Tuning:** Increase `n_trials` in Optuna for a more exhaustive search, or explore different sampling strategies.
*   **Real-World Data:** The ultimate improvement would be to train and validate these models on actual wearable sensor data and corresponding ground-truth hydration measures (e.g., plasma osmolality, USG).
*   **Feature Engineering on Real Data:** With real sensor data, advanced features (HRV metrics, activity patterns over time) could be engineered.
*   **Explore Time-Series Models:** If continuous time-series sensor data becomes the primary input, LSTMs or other sequence models could be explored to capture temporal dependencies more effectively.
*   **Personalized Models:** Develop models that adapt to individual physiological baselines and responses, rather than a general model.
*   **Fatigue Prediction:** Extend the prototype to predict fatigue levels, potentially using similar input features or incorporating sleep data, cognitive load, etc.
*   **Ensemble Methods:** Combine predictions from XGBoost and Random Forest for potentially more robust results.
