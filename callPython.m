% callPython.m — Benchmark MATLAB classifiers against scikit-learn classifiers
% across several sklearn-bundled datasets.
%
% MATLAB drives the workflow:
%   1. Ask Python (via compare.py) for the list of dataset keys.
%   2. For each dataset, ask Python for the same train/test split sklearn uses.
%   3. Train a fleet of MATLAB classifiers on that split.
%   4. Hand all MATLAB predictions back to Python, which trains a matching
%      sklearn fleet, builds skore EstimatorReports / ComparisonReports, and
%      renders a multi-section HTML site.

disp('Calling Python from MATLAB in a GitHub Action!');
disp(pyenv);

cmp = py.importlib.import_module('compare');
py.importlib.reload(cmp);

% Resolve the output directory against GITHUB_WORKSPACE so the site lands at
% the repo root regardless of where matlab-actions/run-command set the CWD.
repo_root = getenv('GITHUB_WORKSPACE');
if isempty(repo_root)
    repo_root = pwd;
end
site_dir = fullfile(repo_root, 'site');

dataset_keys = cell(cmp.dataset_keys());
all_preds = py.dict();

for i = 1:numel(dataset_keys)
    key = dataset_keys{i};
    fprintf('\n=== %s ===\n', char(key));

    split = cmp.get_split(key);
    X_train = double(split{1});
    X_test  = double(split{2});
    y_train = double(split{3});
    y_test  = double(split{4});

    matlab_preds = py.dict();

    % fitcecoc — multiclass via error-correcting output codes over linear SVMs.
    mdl = fitcecoc(X_train, y_train);
    yhat = double(predict(mdl, X_test));
    matlab_preds{'fitcecoc'} = py.numpy.array(yhat(:)');
    fprintf('  fitcecoc:        %.4f\n', mean(yhat(:) == y_test(:)));

    % fitctree — single decision tree.
    mdl = fitctree(X_train, y_train);
    yhat = double(predict(mdl, X_test));
    matlab_preds{'fitctree'} = py.numpy.array(yhat(:)');
    fprintf('  fitctree:        %.4f\n', mean(yhat(:) == y_test(:)));

    % TreeBagger — random forest. predict() returns a cell array of strings,
    % so str2double brings them back to numeric class labels.
    mdl = TreeBagger(50, X_train, y_train);
    yhat = str2double(predict(mdl, X_test));
    matlab_preds{'TreeBagger(50)'} = py.numpy.array(yhat(:)');
    fprintf('  TreeBagger(50):  %.4f\n', mean(yhat(:) == y_test(:)));

    % fitcknn — k-nearest neighbours.
    mdl = fitcknn(X_train, y_train, 'NumNeighbors', 5);
    yhat = double(predict(mdl, X_test));
    matlab_preds{'fitcknn(k=5)'} = py.numpy.array(yhat(:)');
    fprintf('  fitcknn(k=5):    %.4f\n', mean(yhat(:) == y_test(:)));

    all_preds{key} = matlab_preds;
end

fprintf('\nRendering site to %s ...\n', site_dir);
index_path = cmp.render_all(all_preds, pyargs('output_dir', site_dir));
fprintf('Skore HTML report written to: %s\n', char(index_path));
