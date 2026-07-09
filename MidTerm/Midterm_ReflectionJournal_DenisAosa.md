# Mid-Term Reflective Journal: Exploratory Data Analysis & Preprocessing
**Student Name**: Denis Aosa (Denis Bosire)  
**Course**: ITAI 1371 - Introduction to Machine Learning  
**Group**: ML_1371_12321 Group 4  

---

## 1. Concept Breakthroughs & Deep Learning Insights
Collaborating on the Mid-Term project using the **Airbnb Open Data** dataset has consolidated my understanding of data cleaning and preprocessing in a real-world, large-scale context. Working with a dataset of over 100,000 listings presented unique challenges compared to the smaller datasets we analyzed in early labs. 

My primary breakthrough during this project was understanding the importance of **feature scaling and log transformations** on skewed numerical distributions:
- Prior to log-transformation, the `number_of_reviews` feature exhibited extreme right-skewness (skewness of 3.86). By applying `np.log1p(df['number_of_reviews'])`, we reduced the skewness to 0.72, transforming the heavily compressed distribution into a normal-like curve. This is crucial because many ML algorithms assume normally distributed inputs, and highly skewed features can cause gradient-based models to assign disproportionate weight to tail values.
- I also deepened my understanding of **multicollinearity** (the dummy variable trap). When one-hot encoding variables like `room_type` or `cancellation_policy`, failing to set `drop_first=True` creates perfectly correlated columns, which undermines the mathematical assumptions of linear and logistic models.

## 2. Methodology & Implementation (Tasks 1-5)
In this group project, I was responsible for completing the core preprocessing steps inside the joint notebook `app.ipynb` (Tasks 1-5):
1. **One-Hot Encoding (Task 1)**: I implemented `pd.get_dummies()` on the categorical columns `room_type` and `cancellation_policy` with `drop_first=True` and `dtype=int` to generate binary indicators suitable for machine learning algorithms.
2. **Class Imbalance Mitigation (Task 2)**: I handled target class distribution by downsampling the majority class (`instant_bookable` = False) without replacement using Scikit-Learn's `resample` utility. This ensures the model does not suffer from accuracy paradox (where it predicts the majority class and ignores the minority class).
3. **High-Cardinality and Ordinal Encoding (Task 3)**: I mapped the `neighbourhood_group` categories to category codes (`.cat.codes`) to preserve group identity in a numeric format, and converted `host_identity_verified` into numeric category codes.
4. **Metadata Filtering & Output Generation (Task 4)**: I constructed an exclusion filter to drop non-predictive metadata (such as `id`, `name`, `host_id`, `host_name`, `license`, `house_rules`, and `last_review`) and exported the final clean dataframe to `DATA/Airbnb_Cleaned_Data.csv` (70,896 rows, 21 columns).
5. **Data Partitioning (Task 5)**: I implemented the train-test split, reserving 70% of the dataset for training (`X_train` shape: 49,627 rows) and 30% for testing (`X_test` shape: 21,269 rows), using a fixed random state (`random_state=42`) for reproducibility.

## 3. Real-World Applications & Industry Connections
Predicting listing bookability is highly relevant to the **real estate and hospitality technology sectors**:
- **Search Optimization**: Vacation rental platforms (like Airbnb and Vrbo) prioritize instantly bookable listings in their search rankings because they reduce user friction and increase transaction conversion. By understanding which features (e.g., pricing ratios, neighbourhood groups, host verification) correlate with instant bookability, platforms can design better recommendations.
- **Dynamic Pricing & Revenue Management**: Hosts can utilize the predictive models to understand if adjusting their prices or cancellation policies makes their listing competitive enough to enable instant booking without increasing cancellation risks.

## 4. Challenges Faced & Solutions
The most difficult challenge was managing missing values across complex textual and numerical columns without losing data rows. Imputing values using column medians and modes was essential to maintain the dataset's size (above the 2000-row minimum required by the professor). 

Additionally, we had to coordinate code changes across the team using Git and GitHub pull requests. To avoid merge conflicts, we established clear cell boundaries and used a centralized branch comparison before merging my changes into the master branch.
