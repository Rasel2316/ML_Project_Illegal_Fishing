import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import joblib

class Classification:
    def __init__(self, path='fishing_imputed.csv', clf_opt='lr', no_of_selected_features=None):
        self.path = path
        self.clf_opt = clf_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features is not None:
            self.no_of_selected_features = int(self.no_of_selected_features)
        self.trained_models = []

    def classification_pipeline(self):
        if self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = LogisticRegression(solver='liblinear', class_weight='balanced')
            be2 = DecisionTreeClassifier(max_depth=50)
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            clf_parameters = {
                'clf__base_estimator': (be1, be2),
                'clf__random_state': (0, 10),
            }
        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear', class_weight='balanced')
            clf_parameters = {
                'clf__random_state': (0, 10),
            }
        elif self.clf_opt == 'rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None, class_weight='balanced')
            clf_parameters = {
                'clf__criterion': ('entropy', 'gini'),
                'clf__n_estimators': (30, 100),
                'clf__max_depth': (10, 50, 200),
            }
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced', probability=True)
            clf_parameters = {
                'clf__C': (0.5, 5),
                'clf__kernel': ['linear'],
            }
        elif self.clf_opt == 'dt':  # Add Decision Tree classifier
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(max_depth=50, class_weight='balanced')
            clf_parameters = {
                'clf__criterion': ('entropy', 'gini'),
                'clf__splitter': ('best', 'random'),
                'clf__min_samples_split': (2, 5, 10),
                'clf__min_samples_leaf': (1, 2, 4),
                'clf__random_state': (0, 10),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)
        return clf, clf_parameters

    def get_class_statistics(self, labels):
        class_statistics = Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t' + str(item) + '\t\t\t' + str(class_statistics[item]))

    def get_data(self):
        reader = pd.read_csv(self.path)
        data = reader.drop(['target'], axis=1)
        labels = reader['target']
        self.get_class_statistics(labels)
        training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, test_size=0.2,
                                                                                        random_state=42, stratify=labels)
        return training_data, validation_data, training_cat, validation_cat

    def classification(self):
        training_data, validation_data, training_cat, validation_cat = self.get_data()
        clf, clf_parameters = self.classification_pipeline()
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(k=self.no_of_selected_features)),
            ('clf', clf),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(training_data, training_cat):
            X_train, X_test = training_data.iloc[train_index], training_data.iloc[test_index]
            y_train, y_test = training_cat.iloc[train_index], training_cat.iloc[test_index]

            grid = GridSearchCV(pipeline, clf_parameters, scoring='f1_macro', cv=2)
            with tqdm(total=2) as pbar:
                pbar.set_description("Grid Search Progress")
                grid.fit(X_train, y_train)
                pbar.update(1)
                clf = grid.best_estimator_
                pbar.update(1)

                # Save the trained model
                self.trained_models.append(clf)
                joblib.dump(clf, f"{self.clf_opt}_model_{len(self.trained_models)}.joblib")
                print(f"Saved the trained {self.clf_opt} model to {self.clf_opt}_model_{len(self.trained_models)}.joblib")

                # Evaluation on the validation set
                predicted = clf.predict(X_test)

                print('\n *************** Confusion Matrix ***************  \n')
                print(confusion_matrix(y_test, predicted))

                class_names = list(Counter(y_test).keys())
                class_names = [str(x) for x in class_names]

                print('\n ##### Classification Report ##### \n')
                print(classification_report(y_test, predicted, target_names=class_names))

                pr = precision_score(y_test, predicted, average='macro')
                print('\n Precision:\t' + str(pr))

                rl = recall_score(y_test, predicted, average='macro')
                print('\n Recall:\t' + str(rl))

                fm = f1_score(y_test, predicted, average='macro')
                print('\n F1-Score:\t' + str(fm))

        # Save the best model
        best_model = clf
        joblib.dump(best_model, f"{self.clf_opt}_best_model.joblib")
        print(f"Saved the best {self.clf_opt} model to {self.clf_opt}_best_model.joblib")