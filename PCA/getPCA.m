function [prinComp,evals]= getPCA(data)
% subtract the cross-trial mean
dataCT = data - mean(data,2);
% calculate the covariance matrix
covMat = dataCT'*dataCT/size(data,2);
% get eigenvalues and eigenvectors
[evects, evals] = eig(covMat);
ev = evals;
% sort both eigenvalues and eigenvectors from largest to smallest weighting
[~,sortIdx] = sort(diag(evals),'descend');
evects = evects(:,sortIdx);
% project firing rate data onto the newly derived basis
prinComp = dataCT*evects;
% normalisation
prinComp = prinComp./sqrt(sum(prinComp.^2));
% just getting the eigenvalues and not all the other zeros of the diagonal
% matrix
evalsDiag = diag(evals);
evals = diag(evalsDiag(sortIdx));
end
