# Python 3.7x
import numpy as np
import pandas as pd

# Some Udemy course import syntax is outddated, needed google/StackOverflow to fix
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.linear_model           import LogisticRegression
from sklearn.metrics                import classification_report, confusion_matrix
from sklearn.model_selection        import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes            import GaussianNB
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.preprocessing          import LabelEncoder, MinMaxScaler
from sklearn.svm                    import SVC
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.utils                  import shuffle

# import warnings filter
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# StackOverflow convergence warning suppresion
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class DataLoader:
    # Class loads the input data, cleans it and prepares a test/train set split
    def __init__(self,filename):
        self.filename = filename
        # Load file, use validation on columns
        self.initialdata = self.initialfileload(self.filename)
        # Apply some specific optimisations, e.g. categorial encoding, normalisation etc.
        self.cleandata = self.dataprep(self.initialdata)
        self.splitdataset(self.cleandata)
        return

    def initialfileload(self,filename):
        # Read the initial xls file, use the validator functions on each column
        initialdata = pd.read_excel(self.filename, converters={
                                                    'deg-malig':self.validate_malig,
                                                    'age':self.validate_age,
                                                    'menopause':self.validate_menopause,
                                                    'tumor-size':self.validate_tumorsize,
                                                    'inv-nodes':self.validate_stripdates,
                                                    'tumor-size':self.validate_stripdates,
                                                    'breast-quad':self.validate_bq
                                                    })
        print("INFO: XLS file loaded and coarse cleaned.")
        return initialdata

    def validate_age(self,cell):
        # Age is given as a range of 9 years e.g. 40-49. We are just going to use the lower-bound figure
        operand = str(cell)
        return operand[0:2]

    def validate_stripdates(self,cell):
        # tumor-size,inv-nodes columns are corrupt and has dates in, so we NaN them by looking for a colon
        operand = str(cell)
        if ":" in operand:
            return np.nan
        else:
            return operand.split("-")[1]

    def validate_bq(self,cell):
        allowed = [
                "left_up",
                "left_low",
                "right_up",
                "right_low",
                "central",
        ]
        if cell in allowed:
            return cell
        else:
            return np.nan

    def validate_menopause(self,cell):
        allowed = [
                "premeno",
                "ge40",
                "lt40",
        ]
        if cell in allowed:
            return cell
        else:
            return np.nan

    def validate_tumorsize(self,cell):
        # We'll just take the upper end of the tumour size; we don't lose anything by making numeric (all ranges same)
        working = str(cell)
        x = working.split("-")
        return x[1]

    def validate_malig(self,cell):
        operand = str(cell)
        if operand in ["1","2","3"]:
            # Append some text here; shortcut to some easy column names when we later get_dummies from this column
            output = "stage" + operand
            return output
        else:
            return np.nan

    def dataprep(self,frame):
        # Function to clean up the specific dataset we have been given to work on
        # Would need some more generalising for an abitrary submission.
        initialdata =  frame
        # Some specific use-case here, might generalise to a converter for node-caps
        indexNames = initialdata[ initialdata['node-caps'] == "?"].index
        initialdata.drop(indexNames, inplace=True)
        # Some methods here to turn string booleans into ones and zeros for the obvious columns
        lb = LabelEncoder()
        initialdata['irradiat'] =     lb.fit_transform(initialdata['irradiat'])
        initialdata['breast'] =       lb.fit_transform(initialdata['breast'])
        initialdata['node-caps'] =    lb.fit_transform(initialdata['node-caps'])
        initialdata['Class'] =        lb.fit_transform(initialdata['Class'])
        # Encode categorical features, noting we may first have done some string processing on these upon import
        menopause   = pd.get_dummies(initialdata['menopause'],    drop_first=True)
        breast_quad = pd.get_dummies(initialdata['breast-quad'],  drop_first=True)
        deg_malig   = pd.get_dummies(initialdata['deg-malig'],    drop_first=True)
        # Reconverge categorical columns into main dataset
        finaldata = pd.concat([initialdata,menopause,breast_quad,deg_malig],axis=1)
        # Drop the original categorical columns, no longer needed
        finaldata.drop(['breast-quad','menopause','deg-malig'],axis=1,inplace=True)
        # Want to normalise the age and tumor size [sic] columns to be between zero/one or some models won't work (e.g. SVM)
        scaler = MinMaxScaler()
        finaldata[['age', 'tumor-size']] = scaler.fit_transform(finaldata[['age', 'tumor-size']])
        # Drop all rows that contain a NaN, noting we've populated some of ourselves these to cope with dirty data
        finaldata.dropna(inplace=True)
        # How we looking now?
        print("\nCleaned data peek:")
        print(finaldata.head())
        # Dump a cleaned dataset to disk. Unclear how much more mangling of this data is possible?!
        outputfilename = "cleaned_" + self.filename[0:-3] + "csv"
        finaldata.to_csv(outputfilename, index = False)
        print("INFO: Data cleaned including categorical expansion and normalisation.")
        print("INFO: Cleaned data dumped to disk in CSV here: " + outputfilename)
        return finaldata

    def splitdataset(self,cleandata):
        # Generically generate some training/test data splits, per Udemy course.
        # Class is our category we want to predict, 1=RecurrenceEvents 0=NoRecurrenceEvents
        # cleandata = shuffle(cleandata)
        self.X = cleandata.drop('Class', axis=1)
        self.Y = cleandata['Class']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.3,random_state=100)
        print("INFO: X_train,X_test,Y_train,Y_test generated.")
        return



def attempt1_LR(bc_data):
    '''
    ////////////////////////////// Logistic Regression Classifier ///////////////////////////////////
    Uses basic syntax from Udemy course video example.
    Now using liblinear because it's noted for small datasets in sklearn docs
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Previously lbfgs solver was not converging with default number of iterations.
    '''
    LR_model = LogisticRegression(solver='liblinear')
    LR_model.fit(bc_data.X_train,bc_data.Y_train)
    print("\n")
    print("////////////////////// Logistic Regression Attempt //////////////////////")
    predictions = LR_model.predict(bc_data.X_test)

    print(classification_report(bc_data.Y_test,predictions))
    cm = confusion_matrix(bc_data.Y_test,predictions)
    print(cm)

    '''
    Gives:
                  precision    recall  f1-score   support

           0       0.90      0.92      0.91        49
           1       0.43      0.38      0.40         8

    accuracy                           0.84        57
   macro avg       0.66      0.65      0.65        57
weighted avg       0.83      0.84      0.84        57
    '''

def attempt2_LR_randomised(bc_data):
    '''
    ////////////////////////// Randomised Search Hyperparameters on LR Classifier //////////////////
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    '''

    param_space =   {
                    "solver":["liblinear","newton-cg","lbfgs","sag","saga"],
                    "max_iter":[10,50,100,200,500,1000,10000,100000,1000000,10000000],
                    }

    # Form a logistic regression object
    LR_model = LogisticRegression()
    # Set up a KFold iterator. Lots of literature suggests sticking with 10 as a good default.
    kf = KFold(n_splits=10, random_state=5)
    # Set up randomised search mechanism for params. Default iter is 10x.
    rand_grid = RandomizedSearchCV(LR_model,param_space,cv=kf)
    # Execute the model and score it
    rand_grid.fit(bc_data.X_train,bc_data.Y_train)
    accuracy = rand_grid.score(bc_data.X_test,bc_data.Y_test)
    print("\nRandomised Logistic Regression results:")
    print("Accuracy: " +  str(accuracy))
    print("Best params: " + str(rand_grid.best_params_))
    print("\n")

    '''
    Accuracy: 0.8421052631578947
    Best params: {'solver': 'liblinear', 'max_iter': 100}

    Accuracy: 0.8421052631578947
    Best params: {'solver': 'saga', 'max_iter': 50}

    Accuracy: 0.8421052631578947
    Best params: {'solver': 'sag', 'max_iter': 100000}

    Conclusion: Dataset is too small and/or so optimised through data preparation step that changing params of
    the LR solver is not assisting here at all! Get lots of convergence warnings.
    '''


def attempt4_manymodels_randomise(bc_data, force="Grid"):
    '''
    Bringing it all together here. Stacking models into an iterable, randomized search, grid search and KFold CV
    Background reading:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
    https://martin-thoma.com/comparing-classifiers/
    '''

    final_go = []
    final_go.append({
        "name":"Logistic Regression     ",
        "model": LogisticRegression(),
        "force_params": {"solver":["liblinear","newton-cg","lbfgs","sag","saga"],
                        "max_iter":[10,50,100,200,500,1000,10000,100000,1000000,10000000],
                        },
        })
    final_go.append({
        "name":"K-Nearest Neighbours    ",
        "model": KNeighborsClassifier(),
        "force_params": {"n_neighbors":[2,3,4,5,6,7,8,9,10]},
        })
    final_go.append({
        "name":"Decision Tree           ",
        "model": DecisionTreeClassifier(),
        "force_params": {"criterion":["gini","entropy"],
                        "splitter":["best","random"]},
        })
    final_go.append({
        "name":"Support Vector Machine ",
        "model": SVC(),
        "force_params": {"C":[0.1,0.2,0.3,0.5,1,1.5,1.9,2,2.1,2.4,2.5,2.6,3,10,100,1000],
                        "gamma":[1,0.1,0.01,0.05,0.001,0.002,0.0001],
                        },
        })
    final_go.append({
        "name":"Linear Discrim Analysis ",
        "model": LinearDiscriminantAnalysis(),
        "force_params": {"solver":["svd","lsqr","eigen"],
                        },
        })
    final_go.append({
        "name":"Gaussian Naive Bayes    ",
        "model": GaussianNB(),
        "force_params": {},
        # SKLearn features almost no way of tuning this without adding/removing columns.
        # Assumes all features are independent which might be hurting the results here, definitely not performing...
        })

    results = []

    if force == "None":
        print("\nDEFAULT PARAMS SEARCH RESULTS:")
        # No grid search or randomised search, just bulk run models with no tuning.
        for item in final_go:
        	kf = KFold(n_splits=10, random_state=5)
        	crossvalidation = cross_val_score(item['model'], bc_data.X, bc_data.Y, cv=kf, scoring='accuracy', n_jobs=None)
        	results.append(crossvalidation)
        	str_out = "%s \t\t %f \t (%f)" % (item["name"], crossvalidation.mean(), crossvalidation.std())
        	print(str_out)

    elif force == "Randomised":
        # Randomised tuing of models.
        print("\nRANDOMISED SEARCH RESULTS:")
        for item in final_go:
            kf = KFold(n_splits=10, random_state=5)
            rand_grid = RandomizedSearchCV(item['model'],item['force_params'],cv=kf)
            rand_grid.fit(bc_data.X_train,bc_data.Y_train)
            accuracy = rand_grid.score(bc_data.X_test,bc_data.Y_test)
            str_out = "%s \t\t %f \t (%s)" % (item["name"], accuracy, rand_grid.best_params_)
            print(str_out)

    elif force == "Grid":
        # Gride search tuning of models
        print("\nGRID SEARCH RESULTS:")
        for item in final_go:
            kf = KFold(n_splits=10, random_state=5)
            search_grid = GridSearchCV(item['model'],item['force_params'],cv=kf)
            search_grid.fit(bc_data.X_train,bc_data.Y_train)
            accuracy = search_grid.score(bc_data.X_test,bc_data.Y_test)
            str_out = "%s \t\t %f \t (%s)" % (item["name"], accuracy, search_grid.best_params_)
            print(str_out)


    # NOTES:
    # Dataset doesn't seem big enough to use kfold very well? Is k-1 fold statistically big enough to matter?
    # cv=kfold doesn't need to be passed the KFold object, we could just do this with an integer. Left for completeness but see: https://scikit-learn.org/stable/modules/cross_validation.html 3.1.1
    # Reporting mean accuracy at the moment, could try out scoring='f1_macro'; literature suggests this will be quite different since dataset is not balanced across classes
    # Best attempts converge towards 0.859649. Is that a fundamental limitation of the data or something wrong with prep/models.

if __name__ == "__main__":

    jhub = DataLoader("breast-cancer.xls")
    attempt1_LR(jhub)
    attempt2_LR_randomised(jhub)
    #attempt3_manymodels(jhub) Deprecated, expanded into monster attempt4 and generalised.
    attempt4_manymodels_randomise(jhub, force="None")
    attempt4_manymodels_randomise(jhub, force="Randomised")
    attempt4_manymodels_randomise(jhub, force="Grid")
