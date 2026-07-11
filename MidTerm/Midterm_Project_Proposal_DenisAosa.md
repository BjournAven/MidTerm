# Mid-Term Project Proposal: Airbnb Instant Bookability Prediction
**Student Name**: Denis Aosa (Denis Bosire)  
**Course**: ITAI 1371 - Introduction to Machine Learning  
**Group**: ML_1371_12321 Group 4  

---

### 1. Dataset Description
The **Airbnb Open Data** dataset contains approximately 102,599 rows of New York City Airbnb listing activities and characteristics. Key attributes include geographic location (`neighbourhood_group`, `neighbourhood`), listing specifics (`room_type`, `construction_year`), economic details (`price`, `service_fee`), hosting attributes (`host_identity_verified`), user metrics (`number_of_reviews`, `reviews_per_month`), and booking constraints (`cancellation_policy`, `minimum_nights`, `instant_bookable`). 

### 2. Objectives and Target Outcome
The primary objective of this project is to develop a predictive machine learning classifier to determine whether an Airbnb listing is **instantly bookable** (`instant_bookable` = True/False) based on its operational, structural, and pricing properties. Predicting instant bookability is highly valuable for both hosts and the platform, as it influences booking conversion rates, search ranking visibility, and user booking friction.

### 3. Proposed Methodology
1. **Data Preprocessing & Cleaning**:
   - Resolve spatial typos (e.g., merging "brookln" into "Brooklyn", "manhatan" into "Manhattan").
   - Impute missing numerical features (e.g., filling price/fee NaNs with medians) and categorical features (e.g., mode imputing room type).
   - Drop non-predictive metadata (such as `id`, `name`, `host_id`, `host_name`, `license`) to avoid over-parameterization.
2. **Feature Engineering**:
   - Create `fee_to_price_ratio` as an indicator of pricing overhead.
   - Apply Min-Max scaling to the `construction_year` attribute to bound its range to $[0, 1]$.
   - Apply a log transform (`np.log1p`) to the highly skewed `number_of_reviews` feature.
3. **Encoding & Dimensionality Management**:
   - Perform one-hot encoding on categorical variables like `room_type` and `cancellation_policy` (using `drop_first=True` to prevent multicollinearity).
   - Convert high-cardinality groups (like `neighbourhood_group`) to numerical codes using category mapping.
4. **Addressing Class Imbalance**:
   - The target class `instant_bookable` shows a relatively balanced raw distribution, but any slight imbalance will be managed via majority class downsampling using Scikit-Learn's `resample` module to ensure unbiased classification thresholds.
5. **Validation Plan**:
   - Split the clean data into a $70\%$ training set and $30\%$ test set (with a fixed `random_state=42`).
   - Train baseline classifiers (e.g., Logistic Regression, Decision Trees) and evaluate using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
