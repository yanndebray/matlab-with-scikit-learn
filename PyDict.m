classdef PyDict
    %PYDICT  A Python dict assembled on the MATLAB side.
    %   Filled in locally and converted only when it crosses the bridge, so
    %   building one is free. Both index styles work:
    %
    %       preds = py.dict();
    %       preds{'fitctree'} = yhat;      % brace, as the MATLAB driver writes
    %       preds('fitctree') = yhat;      % paren
    %       yhat = preds{'fitctree'};
    %
    %   Keys may be char or numeric. pyargs returns a PyDict flagged as
    %   keyword arguments, which py.call splats into **kwargs. Dicts coming
    %   back from Python arrive as PyDict too.
    %
    %   Three RunMat workarounds shape this class (see RUNMAT.md): properties
    %   carry no defaults, because `= {}` silently becomes a double and
    %   `= false` panics the compiler; every cell is read into a local before
    %   being indexed or grown; and nothing switches on a char.

    properties
        keylist
        vallist
        iskw
    end

    methods
        function obj = PyDict(isKeywords)
            obj.keylist = {};
            obj.vallist = {};
            obj.iskw = 0;
            if nargin >= 1
                obj.iskw = double(logical(isKeywords));
            end
        end

        function obj = insert(obj, key, value)
            %INSERT  Set KEY to VALUE, replacing any existing entry.
            ks = obj.keylist;
            vs = obj.vallist;
            at = 0;
            for i = 1:numel(ks)
                if isequal(ks{i}, key)
                    at = i;
                    break;
                end
            end
            if at > 0
                vs{at} = value;
            else
                ks{end + 1} = key;
                vs{end + 1} = value;
            end
            obj.keylist = ks;
            obj.vallist = vs;
        end

        function value = get(obj, key)
            %GET  Look up KEY, erroring if it is absent.
            ks = obj.keylist;
            vs = obj.vallist;
            for i = 1:numel(ks)
                if isequal(ks{i}, key)
                    value = vs{i};
                    return;
                end
            end
            if ischar(key)
                error('py:PyDict:missingKey', 'no entry for %s', key);
            end
            error('py:PyDict:missingKey', 'no such entry');
        end

        function tf = has(obj, key)
            %HAS  True if KEY is present.
            ks = obj.keylist;
            tf = false;
            for i = 1:numel(ks)
                if isequal(ks{i}, key)
                    tf = true;
                    return;
                end
            end
        end

        function k = keys(obj)
            %KEYS  Cell array of keys, in insertion order.
            k = obj.keylist;
        end

        function v = vals(obj)
            %VALS  Cell array of values, in insertion order.
            v = obj.vallist;
        end

        function n = count(obj)
            %COUNT  Number of entries.
            ks = obj.keylist;
            n = numel(ks);
        end

        function tf = iskwargs(obj)
            %ISKWARGS  True for the dict pyargs returns.
            tf = logical(obj.iskw);
        end

        function obj = subsasgn(obj, s, value)
            if numel(s) ~= 1 || ~any(strcmp(s(1).type, {'{}', '()'}))
                error('py:PyDict:assign', ...
                    'assign into a PyDict with d{key} = value');
            end
            subs = s(1).subs;
            if ~iscell(subs)
                subs = {subs};
            end
            if numel(subs) ~= 1
                error('py:PyDict:assign', 'a PyDict takes one key at a time');
            end
            obj = insert(obj, subs{1}, value);
        end

        function varargout = subsref(obj, s)
            if strcmp(s(1).type, '{}') || strcmp(s(1).type, '()')
                subs = s(1).subs;
                if ~iscell(subs)
                    subs = {subs};
                end
                if numel(subs) ~= 1
                    error('py:PyDict:index', 'a PyDict takes one key');
                end
                varargout{1} = get(obj, subs{1});
            elseif strcmp(s(1).type, '.')
                name = char(s(1).subs);
                if strcmp(name, 'keys')
                    varargout{1} = obj.keylist;
                elseif strcmp(name, 'vals')
                    varargout{1} = obj.vallist;
                elseif strcmp(name, 'count')
                    ks = obj.keylist;
                    varargout{1} = numel(ks);
                else
                    error('py:PyDict:field', ...
                        'PyDict has no %s - use d{key}, keys(d), vals(d)', name);
                end
            else
                error('py:PyDict:index', 'unsupported indexing');
            end
        end

        function disp(obj)
            ks = obj.keylist;
            vs = obj.vallist;
            if logical(obj.iskw)
                fprintf('  <python keyword arguments, %d entries>\n', numel(ks));
            else
                fprintf('  <python dict, %d entries>\n', numel(ks));
            end
            for i = 1:numel(ks)
                key = ks{i};
                if ~ischar(key)
                    key = mat2str(key);
                end
                value = vs{i};
                if ischar(value)
                    shown = value;
                elseif isnumeric(value) || islogical(value)
                    shown = sprintf('%s %dx%d', class(value), ...
                        size(value, 1), size(value, 2));
                else
                    shown = class(value);
                end
                fprintf('    %s: %s\n', key, shown);
            end
        end
    end
end
