import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(n_class=2, return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier(tmp_folder='thesis_tmp',
                                                          output_folder='thesis_out',
                                                          delete_tmp_folder_after_terminate=False,
                                                          delete_output_folder_after_terminate=False,
                                                          shared_mode=True)

identifier = (1, 5)
automl._automl = automl.build_automl()
automl.show_models()
automl._automl._classes = automl._automl._process_target_classes(y)
automl._automl.models_ = automl._automl._backend.load_models_by_identifiers([identifier])
print(automl._automl.models_[identifier].predict(X_test))
