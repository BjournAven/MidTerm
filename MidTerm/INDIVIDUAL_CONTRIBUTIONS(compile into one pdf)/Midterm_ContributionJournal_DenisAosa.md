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
* **Data Ingestion**: Made a code cell as a data loading block, wrapped it with error handling
* **Missing Value Imputation** Made a zero null imputation that replaced missing values with modes, number gaps with medians, and weird texxt strings with a placeholder
* **Logarithmic Transformation** Fixed distribution assymetry using dynamic logarithmic transformations
* **REFLECTION** Made individual reflective journal
* **Feature Engineering** Pruned pointless zero value pricing rows and made a feature profile `fee_to_price_ratio` to point out crosras column pricing interactions


### Victoria Simmons - Contributions
- helped with feature scaling by using min-max scaling function to map the datasets timeline
- Feature clean up : automated column key normalization to snake_case, made an automated text parsing routine to convert character literals to floats and remove them from currency fields.
- Wrote individual reflection journal
- Participated in code review
- Helped with documentations
- Participated and contributed in active discussion


### Mary Balemba and Raymond Hayes- Contributions
- Reviewed the team's preprocessing work to ensure it met project requirements.
- Verified the Airbnb dataset was appropriate and contained enough records.
- Checked that missing values, duplicates, and invalid records were handled correctly.
- Confirmed categorical features were encoded and scaling/normalization was applied where appropriate.
- Reviewed the engineered feature to ensure it was reasonable.
- Verified the dataset was split into 70% training and 30% testing using Python.
- Reviewed the notebook and documentation for clarity and completeness of the preprocessing steps and project objectives.