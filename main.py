import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn.metrics
import numpy as np

dataset_name = 'arrythmia'
data_str = np.genfromtxt(fname='datasets/' + dataset_name + '.txt', delimiter=',', dtype=str)
features_count = len(data_str[0]) - 1
samples_count = len(data_str)

X_str = data_str[:samples_count, :features_count - 1]
y_str = data_str[:samples_count, features_count]

X_int = LabelEncoder().fit_transform(X_str.ravel()).reshape(*X_str.shape)
y_int = LabelEncoder().fit_transform(y_str.ravel()).reshape(*y_str.shape).astype(int)

X_bin = OneHotEncoder().fit_transform(X_int).toarray().astype(float)

X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X_bin, y_int)
automl = autosklearn.classification.AutoSklearnClassifier(tmp_folder=dataset_name + '_tmp',
                                                          output_folder=dataset_name + '_out',
                                                          delete_tmp_folder_after_terminate=False,
                                                          delete_output_folder_after_terminate=False,
                                                          shared_mode=True)
# automl.fit(X_train, y_train)
# y_hat = automl.predict(X_test)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

identifier = (1, 2)
automl._automl = automl.build_automl()
automl.show_models()
automl._automl._classes = automl._automl._process_target_classes(y_int)
automl._automl.models_ = automl._automl._backend.load_models_by_identifiers([identifier])
loaded_model = automl._automl.models_[identifier]
y_hat = automl._automl.models_[identifier].predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
