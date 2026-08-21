function free(varargin)
%FREE  Drop the pickled Python objects behind one or more handles.
%   py.free(OBJ1, OBJ2, ...) deletes the session copies, so the handles
%   cannot be used again. Handles are cheap files in the session scratch
%   directory; py.session('reset') removes them all at once.

    ids = {};
    for i = 1:numel(varargin)
        obj = varargin{i};
        if ~isa(obj, 'PyObject')
            error('py:free:type', 'py.free expects PyObject handles');
        end
        ref = pyref(obj);
        if strcmp(char(ref.pytag), 'handle')
            ids{end+1} = ref.id;
        end
    end
    if isempty(ids)
        return;
    end

    ctx = py.session('call');
    request = struct();
    request.op = 'free';
    request.ids = ids;
    py.raw(request, ctx);
    py.cleanup(ctx);
end
