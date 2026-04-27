% callPython.m — Train a MATLAB classifier and compare it against
% sklearn's LogisticRegression using skore.
%
% MATLAB drives the whole workflow:
%   1. Ask Python (via compare.py) for the same Iris train/test split sklearn uses.
%   2. Train fitcecoc on that split here in MATLAB.
%   3. Hand both halves and the MATLAB predictions back to compare.py, which
%      builds two skore EstimatorReports and a ComparisonReport.

disp('Calling Python from MATLAB in a GitHub Action!');
disp(pyenv);

cmp = py.importlib.import_module('compare');
py.importlib.reload(cmp);

split = cmp.make_split();
X_train_py = split{1}; X_test_py = split{2};
y_train_py = split{3}; y_test_py = split{4};

% Convert numpy arrays into MATLAB doubles for the Statistics & ML toolbox.
X_train = double(X_train_py);
X_test  = double(X_test_py);
y_train = double(y_train_py);
y_test  = double(y_test_py);

mdl = fitcecoc(X_train, y_train);
y_pred_matlab = predict(mdl, X_test);

fprintf('MATLAB fitcecoc accuracy: %.4f\n', mean(y_pred_matlab(:) == y_test(:)));

% Round-trip predictions through numpy so compare.py sees a 1-D int array.
metrics = cmp.compare(X_train_py, X_test_py, y_train_py, y_test_py, ...
    py.numpy.array(y_pred_matlab(:)));

disp('skore ComparisonReport metrics:');
disp(metrics);
