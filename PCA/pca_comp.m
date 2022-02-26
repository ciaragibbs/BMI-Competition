function [principalComponents,evals] = pca_comp(rates)

    % subtract the cross-trial mean
    rates = rates - mean(rates,1);
    % calculate the covariance matrix
    cov_matrix = (1/size(rates,1))*rates'*rates;
    % get eigenvalues and eigenvectors
    [evects,evals] = eig(cov_matrix);
    % sort both eigenvalues and eigenvectors from largest to smallest
    % weighting
    [evals, sortIdx] = sort(evals,'descend');
    evects = evects(sortIdx);
    % project data onto the new basis
    principalComponents = rates*evects;
   
end