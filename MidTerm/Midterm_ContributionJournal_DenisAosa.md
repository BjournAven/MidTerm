# Contribution Journal: Mid-Term Project
**Course**: ITAI 1371 - Introduction to Machine Learning  
**Group**: ML_1371_12321 Group 4  
**Date**: July 5, 2026  

---

## Group Members
* Denis Aosa (Denis Bosire)
* Aven Gilbert
* Victoria Simmons (Tori)
* Mary Balemba
* Raymond Hayes

---

## Individual Contributions for Mid-Term Project

### Denis Aosa — Contributions
* **Jupyter Notebook Implementation**: Manually completed and executed all 5 major preprocessing tasks in the joint notebook `app.ipynb` (Tasks 1-5):
  * **Task 1**: Implemented one-hot encoding on categorical variables (`room_type`, `cancellation_policy`) with `drop_first=True`.
  * **Task 2**: Balanced target variable classes (`instant_bookable`) via majority class downsampling without replacement using `sklearn.utils.resample`.
  * **Task 3**: Converted categorical variables (`neighbourhood_group`, `host_identity_verified`) to category codes.
  * **Task 4**: Cleaned features and dropped non-predictive metadata columns to export the final preprocessed dataset (`Airbnb_Cleaned_Data.csv`).
  * **Task 5**: Implemented Scikit-Learn's `train_test_split` to divide the dataset into 70% training and 30% testing.
* **Documentation & Proposal**: Drafted the dataset description and project proposal document.
* **Reflective Journal**: Authored individual reflective journal detailing learning outcomes.

### Aven Gilbert — Contributions
* **Project Setup**: Initialized the GitHub repository and uploaded raw dataset.
* **Data Integration**: Created the base skeleton of `app.ipynb` and defined the pre-processing tasks and TODO structure.
* **Visualizations**: Added initial visualization code cells and data distributions in the notebook.

### Victoria Simmons, Mary Balemba, Raymond Hayes — Contributions
* **Code Review & Feedback**: Reviewed and validated the pre-processing logic in the notebook.
* **Project Discussions**: Participated in dataset selection and target variable confirmation discussions.
* **Documentation**: Contributed individual reflective notes and finalized submission components.
