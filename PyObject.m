classdef PyObject
    %PYOBJECT  A reference to something living on the Python side.
    %   Four flavours, all produced by the +py functions rather than by hand:
    %       module   an imported module (py.importlib.import_module)
    %       handle   a pickled object in the interop session (any Python
    %                value the bridge could not convert to a MATLAB one)
    %       bound    an attribute of one of the above that turned out to be
    %                callable, so it can be called later
    %       none     Python None (py.none)
    %       matrix   MATLAB data to send without squeezing (py.matrix)
    %
    %   Dot access goes to Python:
    %
    %       f = cmp.dataset_keys;   % bound PyObject
    %       names = f();            % calls it
    %
    %   RunMat cannot route cmp.dataset_keys() — a dotted name followed by
    %   arguments — through a user-defined subsref, so the one-liner form is
    %   py.call(cmp, 'dataset_keys'). See RUNMAT.md.

    properties
        kind
        name
        id
        parent
        data
        pytype
        pyrepr
    end

    methods
        function obj = PyObject(spec, a, b)
            % if/elseif rather than switch throughout this class: RunMat
            % cannot switch on char values. See RUNMAT.md.
            if isa(spec, 'struct')
                tag = char(spec.pytag);
                obj.kind = tag;
                if strcmp(tag, 'module')
                    obj.name = char(spec.name);
                elseif strcmp(tag, 'handle')
                    obj.id = char(spec.id);
                    if isfield(spec, 'type')
                        obj.pytype = char(spec.type);
                    end
                    if isfield(spec, 'repr')
                        obj.pyrepr = char(spec.repr);
                    end
                else
                    error('py:PyObject:badSpec', ...
                        'cannot build a PyObject from a %s reference', tag);
                end
                return;
            end

            obj.kind = char(spec);
            if strcmp(obj.kind, 'bound')
                obj.parent = a;
                obj.name = char(b);
            elseif strcmp(obj.kind, 'matrix')
                obj.data = a;
            elseif ~strcmp(obj.kind, 'none')
                error('py:PyObject:badKind', 'unknown kind %s', obj.kind);
            end
        end

        function k = kindof(obj)
            %KINDOF  'module', 'handle', 'bound', 'none', or 'matrix'.
            k = obj.kind;
        end

        function d = payload(obj)
            %PAYLOAD  The MATLAB data carried by a py.matrix wrapper.
            d = obj.data;
        end

        function ref = pyref(obj)
            %PYREF  The wire form of this reference (see pybridge.py).
            ref = struct();
            if strcmp(obj.kind, 'module')
                ref.pytag = 'module';
                ref.name = obj.name;
            elseif strcmp(obj.kind, 'handle')
                ref.pytag = 'handle';
                ref.id = obj.id;
            elseif strcmp(obj.kind, 'none')
                ref.pytag = 'none';
            elseif strcmp(obj.kind, 'bound')
                ref.pytag = 'attr';
                ref.of = pyref(obj.parent);
                ref.name = obj.name;
            elseif strcmp(obj.kind, 'matrix')
                error('py:PyObject:matrixRef', ...
                    'py.matrix values are encoded by py.encode, not pyref');
            else
                error('py:PyObject:badKind', 'unknown kind %s', obj.kind);
            end
        end

        function varargout = subsref(obj, s)
            if strcmp(s(1).type, '.')
                varargout{1} = py.getattr(obj, char(s(1).subs));
            elseif strcmp(s(1).type, '()')
                args = s(1).subs;
                if ~iscell(args)
                    args = {args};
                end
                if strcmp(obj.kind, 'bound')
                    varargout{1} = py.call(obj.parent, obj.name, args{:});
                else
                    varargout{1} = py.call(obj, '', args{:});
                end
            else
                error('py:PyObject:index', ...
                    '%s indexing is not defined for PyObject', s(1).type);
            end
        end

        function disp(obj)
            if strcmp(obj.kind, 'module')
                fprintf('  <python module ''%s''>\n', obj.name);
            elseif strcmp(obj.kind, 'handle')
                if ~ischar(obj.pyrepr)
                    fprintf('  <python %s>\n', obj.pytype);
                else
                    fprintf('  <python %s> %s\n', obj.pytype, obj.pyrepr);
                end
            elseif strcmp(obj.kind, 'bound')
                fprintf('  <python callable ''%s''>\n', obj.name);
            elseif strcmp(obj.kind, 'none')
                fprintf('  <python None>\n');
            elseif strcmp(obj.kind, 'matrix')
                fprintf('  <matlab %dx%d, sent to Python unsqueezed>\n', ...
                    size(obj.data, 1), size(obj.data, 2));
            end
        end
    end
end
