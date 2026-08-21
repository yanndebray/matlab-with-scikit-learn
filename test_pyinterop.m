% test_pyinterop.m — exercise the py.* bridge on RunMat.
%
%   runmat test_pyinterop.m
%
% Covers the round trips the port depends on: scalars, strings, matrices,
% cells, dicts, keyword arguments, attribute access, error propagation, and
% object handles. Prints one line per check and exits non-zero on failure.

passed = 0;
failed = 0;

fprintf('py.* interop tests\n');
fprintf('==================\n\n');

info = py.info();
fprintf('python %s at %s\n', info.version, info.executable);
if isempty(info.numpy)
    fprintf('numpy is missing — install it before running the benchmark\n');
end
fprintf('\n');

np = py.importlib.import_module('numpy');
builtins = py.importlib.import_module('builtins');

% --- scalars and strings -------------------------------------------------
[passed, failed] = check('float round trip', ...
    isequal(py.eval('1.5 + 2.5'), 4), passed, failed);

[passed, failed] = check('int round trip', ...
    isequal(py.call(builtins, 'abs', -7), 7), passed, failed);

[passed, failed] = check('string round trip', ...
    strcmp(py.call(builtins, 'str', 'iris'), 'iris'), passed, failed);

[passed, failed] = check('bool round trip', ...
    islogical(py.eval('1 == 1')) && py.eval('1 == 1'), passed, failed);

[passed, failed] = check('None becomes []', ...
    isempty(py.eval('None')), passed, failed);

[passed, failed] = check('NaN survives JSON', ...
    isnan(py.eval('float(''nan'')')), passed, failed);

% --- matrices ------------------------------------------------------------
A = [1 2 3; 4 5 6];
back = py.call(np, 'transpose', A);
[passed, failed] = check('matrix keeps orientation', ...
    isequal(back, A'), passed, failed);

[passed, failed] = check('big matrix survives (200x50)', ...
    isequal(size(py.call(np, 'transpose', ones(200, 50))), [50 200]), ...
    passed, failed);

v = py.call(np, 'cumsum', [1 2 3 4]);
[passed, failed] = check('vector arrives as a row', ...
    isequal(v, [1 3 6 10]), passed, failed);

[passed, failed] = check('vectors go out as 1-D', ...
    isequal(py.eval('x.ndim', struct('x', [1 2 3])), 1), passed, failed);

[passed, failed] = check('py.matrix keeps 2-D', ...
    isequal(py.eval('x.ndim', struct('x', py.matrix([1 2 3]))), 2), ...
    passed, failed);

[passed, failed] = check('logical round trip', ...
    islogical(py.eval('x > 2', struct('x', [1 2 3 4]))), passed, failed);

% --- containers ----------------------------------------------------------
lst = py.eval('[1, "two", None]');
[passed, failed] = check('list becomes a cell', ...
    iscell(lst) && numel(lst) == 3 && strcmp(lst{2}, 'two'), passed, failed);

[passed, failed] = check('cell goes out as a list', ...
    isequal(py.call(builtins, 'len', {1, 2, 3}), 3), passed, failed);

d = py.eval('{"a": 1, "b": [1, 2]}');
[passed, failed] = check('dict becomes a PyDict', ...
    isa(d, 'PyDict') && count(d) == 2 && isequal(get(d, 'a'), 1), ...
    passed, failed);

out = py.dict();
out{'k'} = 42;
[passed, failed] = check('PyDict goes out as a dict', ...
    isequal(py.eval('d["k"]', struct('d', out)), 42), passed, failed);

[passed, failed] = check('struct goes out as a dict', ...
    isequal(py.eval('s["n"]', struct('s', struct('n', 5))), 5), ...
    passed, failed);

% --- keyword arguments ---------------------------------------------------
[passed, failed] = check('pyargs supplies kwargs', ...
    isequal(py.call(np, 'sum', py.matrix([1 2; 3 4]), pyargs('axis', 0)), ...
            [4 6]), passed, failed);

% --- attributes and callables -------------------------------------------
[passed, failed] = check('data attribute reads', ...
    abs(np.pi - pi) < 1e-12, passed, failed);

f = np.sqrt;                       % callable attribute -> bound PyObject
[passed, failed] = check('bound callable calls', ...
    isa(f, 'PyObject') && isequal(f(16), 4), passed, failed);

% --- handles -------------------------------------------------------------
% A dict subclass still converts (Counter comes back as a PyDict); an
% estimator has no MATLAB equivalent, so it is pickled into the session and
% handed over as a handle.
linear = py.importlib.import_module('sklearn.linear_model');
est = py.call(linear, 'LogisticRegression', pyargs('max_iter', 2000));
[passed, failed] = check('estimator becomes a handle', ...
    isa(est, 'PyObject'), passed, failed);

params = py.call(est, 'get_params');
[passed, failed] = check('handle survives a second call', ...
    isa(params, 'PyDict') && isequal(get(params, 'max_iter'), 2000), ...
    passed, failed);

% --- errors --------------------------------------------------------------
ok = false;
try
    py.eval('1/0');
catch err
    ok = ~isempty(strfind(err.message, 'ZeroDivisionError'));
end
[passed, failed] = check('Python exception becomes a MATLAB error', ok, ...
    passed, failed);

ok = false;
try
    py.importlib.import_module('no_such_module_xyz');
catch err
    ok = ~isempty(strfind(err.message, 'ModuleNotFoundError'));
end
[passed, failed] = check('missing module reports cleanly', ok, passed, failed);

% --- escape hatches and lifecycle ---------------------------------------
[passed, failed] = check('py.exec returns a named variable', ...
    isequal(py.exec('y = x * 2', struct('x', [1 2 3]), 'y'), [2 4 6]), ...
    passed, failed);

[passed, failed] = check('py.none reaches Python as None', ...
    strcmp(py.call(builtins, 'str', py.none()), 'None'), passed, failed);

[passed, failed] = check('py.list wraps a cell', ...
    numel(py.list({1, 2, 3})) == 3, passed, failed);

py.free(est);
ok = false;
try
    py.call(est, 'get_params');
catch err
    ok = ~isempty(strfind(err.message, 'freed'));
end
[passed, failed] = check('a freed handle says so', ok, passed, failed);

py.session('reset');
[passed, failed] = check('the bridge survives a session reset', ...
    isequal(py.eval('2 + 2'), 4), passed, failed);

% --- report --------------------------------------------------------------
fprintf('\n%d passed, %d failed\n', passed, failed);
if failed > 0
    error('py:tests', '%d interop test(s) failed', failed);
end
