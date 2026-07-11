# Mid-Term Reflective Journal: Exploratory Data Analysis & Preprocessing
 
**Course**: ITAI 1371 - Introduction to Machine Learning  
**Group**: ML_1371_12321 Group 4  

---


## Denis Aosa - Reflection

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


## Victora Simmons - Reflection


What I Learned:

Working on this project helped me understand how important preprocessing is in data
science. Even simple tasks like encoding or scaling can make a big difference in how
well a model performs.

• Visualizations Matter
Creating the histogram showed me how much insight you can get from a simple plot.
I could immediately see that Airbnb prices are not evenly distributed and may contain
outliers.

Feature Cleanup: I converted all attributes into standard snake_case. I did this to remove syntax headaches. I made and implemented an automatic text parsing. I used string operations to strip literals from the currency fields. By doing this the model can now use this data that it would've originally ignored.

Scaling Improves Model Performance. I had to prevent the model from assigning weight to features that had larger numerical spans. I made a min max scaling function. This function can map the datasets construction_year timeline into a variance controlled spectrum of [0,1]. I developed a deeper understanding of feature normalization and data distribution alignment.s


Challenges I Faced:

My biggest challenge was getting the dataset to load correctly in Colab.
I had errors with the df variable not being defined because the first cell had indentation
issues. Once I fixed the dataset-loading cell, everything else became much easier.
This project helped me understand how data cleaning, feature engineering, encoding,
and scaling all fit together. I feel more confident working with real datasets and contributing to group projects.


## Mary Balemba
This project helped me understand why preparing data is an important step before
building a machine learning model. Before working on this project, I thought the main
focus was creating the model, but I learned that cleaning and organizing the data first is
just as important.
Going through the notebook helped me see how missing values, duplicate records, and
categorical data were handled using Python instead of making changes manually.
I learned why the dataset should be split into training and testing data before doing the
analysis.
Another thing I learned was how to work with a team using GitHub and review code
together. Looking through the preprocessing steps helped me understand how each
step improves the quality of the dataset and prepares it for later use. Overall, this
project gave me more confidence in reading Python code and understanding the
process of preparing data for machine learning.

## Raymond Hayes
Proper cleaning of data is imperative. Cleaning data such as removing symbols and
special characters from columns. Make sure missing values are included in all relevant
columns Finally, use one-hot coding for the coding of variables. Improper cleaning can
result in a biased model, skewed or inaccurate results and false predictions.

Real world applications and relevance:
This project helped me convert an exercise into real world problem solving and how
machine learning is relevant given proper data input and models. It can assist with
overall strategy in the New York area. Questions answered are:
How to price appropriately, what price point will enable a property to be instantly
bookable.
What features and strategies can be implemented to allow competitiveness with other
short term rentals companies.
Who is your target market, what guest are you trying to attract.

Key Takeaways:
Plotting helps with locating any missing date. What features are important. It also lets
you know if there is skewing or outliers. Data cleaning is probably the most important
step. Visualization using plots creates a better understanding of what the data is trying
to tell you. Large datasets are important in providing accuracy and predictability. On
ongoing issue during this class is the “why” As I dive deeper into machine learning, I
understand the process more and how information is processed and used as predictors
and indicator and can be used in overall business decision making.
