% callPython_runmat.m — the RunMat port of callPython.m.
%
% Same benchmark, same Python side (compare.py), same skore report: for each
% sklearn-bundled dataset, ask Python for the split, train a fleet of MATLAB
% classifiers on it, and hand the predictions back for scoring.
%
% Two things differ from the MATLAB driver:
%
%   * Interop. RunMat has no built-in Python bridge, so this repo ships one
%     in +py/ (see pybridge.py and RUNMAT.md). MATLAB writes
%     cmp.get_split(key); RunMat cannot dispatch a dotted call through a
%     user-defined subsref, so the port writes py.call(cmp, 'get_split', key).
%
%   * Classifiers. RunMat ships fitctree but not fitcecoc, fitcknn, or
%     TreeBagger, so the MATLAB side of the benchmark is currently fitctree
%     plus a plain-MATLAB k-NN (knnPredict.m). The rest of the fleet is the
%     next piece of the port.
%
% Usage:  runmat callPython_runmat.m

disp('Calling Python from RunMat!');
py.info();

cmp = py.importlib.import_module('compare');
cmp = py.importlib.reload(cmp);

% Resolve the output directory against GITHUB_WORKSPACE so the site lands at
% the repo root regardless of where CI set the working directory.
repo_root = getenv('GITHUB_WORKSPACE');
if isempty(repo_root) || ~ischar(repo_root)
    repo_root = pwd;
end
site_dir = fullfile(repo_root, 'site');

dataset_keys = py.call(cmp, 'dataset_keys');
all_preds = py.dict();

for i = 1:numel(dataset_keys)
    key = dataset_keys{i};
    fprintf('\n=== %s ===\n', key);

    split = py.call(cmp, 'get_split', key);
    X_train = split{1};
    X_test  = split{2};
    y_train = split{3};
    y_test  = split{4};

    matlab_preds = py.dict();

    % fitctree — RunMat's own ClassificationTree.
    mdl = fitctree(X_train, y_train(:));
    yhat = double(gather(predict(mdl, X_test)));
    matlab_preds{'fitctree'} = yhat(:)';
    fprintf('  fitctree:      %.4f\n', accuracy(yhat, y_test));

    % knnPredict — stand-in for fitcknn(k=5) while RunMat lacks it.
    yhat = knnPredict(X_train, y_train, X_test, 5);
    matlab_preds{'knn(k=5)'} = yhat(:)';
    fprintf('  knn(k=5):      %.4f\n', accuracy(yhat, y_test));

    all_preds{key} = matlab_preds;
end

fprintf('\nRendering site to %s ...\n', site_dir);
index_path = py.call(cmp, 'render_all', all_preds, pyargs('output_dir', site_dir));
fprintf('Skore HTML report written to: %s\n', index_path);
