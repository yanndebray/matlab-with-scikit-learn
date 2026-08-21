function items = aslist(raw)
%ASLIST  Normalise a jsondecoded JSON array into a cell array.
%   RunMat's jsondecode collapses a JSON array of objects into a cell array,
%   a single-element one into a bare struct, and an empty one into []. This
%   flattens all three into a cell so callers can just loop.

    if isempty(raw)
        items = {};
    elseif iscell(raw)
        items = raw;
    elseif isa(raw, 'struct')
        items = cell(1, numel(raw));
        for i = 1:numel(raw)
            items{i} = raw(i);
        end
    else
        error('py:badWire', 'expected a JSON array of objects, got %s', class(raw));
    end
end
