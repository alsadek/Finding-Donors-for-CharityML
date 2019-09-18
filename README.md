# Machine Learning Engineer Nanodegree
# Supervised Learning
## Project: Finding Donors for CharityML


## Introduction
This repo contains all my work for Project 2 of Udacity's Machine Learning Nanodegree Program. In this project, I applied supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause. I first explored the data to learn how the census data is recorded. Next, I applied a series of transformations and preprocessing techniques to manipulate the data into a workable format. Then I evaluated several supervised learners of my choice on the data(Logistic Regression, Gradient Boosting, Random Forest), and considered which is best suited for the solution. Afterwards, I optimized the model I had selected and presented it as my solution to CharityML. Finally, I explored the chosen model and its predictions under the hood, to see just how well it’s performing when considering the data it’s given. then i extract the Feature Importance which is features provide the most predictive power.
The main supervised learning models used:
  - Logistic Regression
  - Gradient Boosting
  - Random Forest

## Description
In this project, you will apply supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML identify people most likely to donate to their cause. First explore the data to learn how the census data is recorded. Next, you will apply a series of transformations and preprocessing techniques to manipulate the data into a workable format. Then evaluate several supervised learners on the data, and consider which is best suited for the solution. Afterwards, you will optimize the model you've selected and present it as your solution to CharityML. Finally, you will explore the chosen model and its predictions under the hood, to see just how well it's performing when considering the data it's given.

## Project Highlights
This project is designed to get you familiar with: 
  - many supervised learning algorithms available in sklearn.
  - many methods of evaluating just how each model works and performs on a certain type of data.
  - understand exactly when and where a certain algorithm should be used, and when one should be avoided.


Things you will learn by completing this project:
  1. How To apply many supervised learning algorithms 
  2. What each of several supervised learning algorithms accomplishes given a specific dataset.
  3. How to investigate whether a candidate solution model is suitable for the problem.
  4. How to identify when preprocessing is needed, and how to apply it.
  5. How to establish a benchmark for a solution to the problem.

### Install
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code

Template code is provided in the `finding_donors.ipynb` notebook file. You will also be required to use the included `visuals.py` Python file and the `census.csv` dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` is meant to be used out-of-the-box and not intended for students to manipulate. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)
