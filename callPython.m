% callPython.m — Demonstrate calling Python (scikit-learn) from MATLAB.
% Trains a logistic regression on the Iris dataset entirely through Python,
% then converts the resulting accuracy back to a MATLAB double.

disp('Calling Python from MATLAB in a GitHub Action!');
disp(pyenv);

np      = py.importlib.import_module('numpy');
datasets = py.importlib.import_module('sklearn.datasets');
modelsel = py.importlib.import_module('sklearn.model_selection');
linmod   = py.importlib.import_module('sklearn.linear_model');
metrics  = py.importlib.import_module('sklearn.metrics');

iris = datasets.load_iris();
split = modelsel.train_test_split(iris.data, iris.target, ...
    pyargs('test_size', 0.25, 'random_state', int32(0)));
X_train = split{1}; X_test = split{2};
y_train = split{3}; y_test = split{4};

clf = linmod.LogisticRegression(pyargs('max_iter', int32(200)));
clf.fit(X_train, y_train);
y_pred = clf.predict(X_test);

accuracy = double(metrics.accuracy_score(y_test, y_pred));
fprintf('Logistic regression accuracy on Iris (from MATLAB): %.4f\n', accuracy);

assert(accuracy > 0.8, 'Expected accuracy > 0.8 from sklearn.LogisticRegression');
