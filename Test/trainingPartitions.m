function varargout = trainingPartitions(numObservations,splits)
arguments
    numObservations (1,1) {mustBePositive}
    splits {mustBeVector,mustBeInRange(splits,0,1,"exclusive"),mustSumToOne}
end
numPartitions = numel(splits);
varargout = cell(1,numPartitions);
idx = randperm(numObservations);
idxEnd = 0;
for i = 1:numPartitions-1
    idxStart = idxEnd + 1;
    idxEnd = idxStart + floor(splits(i)*numObservations) - 1;
    varargout{i} = idx(idxStart:idxEnd);
end
% Last partition.
varargout{end} = idx(idxEnd+1:end);
end
function mustSumToOne(v)
% Validate that value sums to one.
if sum(v,"all") ~= 1
    error("Value must sum to one.")
end
end









