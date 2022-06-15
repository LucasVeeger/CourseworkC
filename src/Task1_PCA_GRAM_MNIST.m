load ('mnist.mat');

% crop data to x samples
samples = 5000;
feats = 28*28;
digits_train = double(digits_train(:,:,1:samples));
digits_train = double(reshape(digits_train, [28*28,samples])');

C = center_datapoints(digits_train);

% forumla from slides  G=Y'*Y
% normalize directly
gram_norm = (C * C')/samples;


%% 4. Compute eigenvalues and eigenvectors of the covariance matrix

% calculate eigenvals and sort 
[evs_unsorted, l_unsorted]=eig(gram_norm);
[l, idx]=sort(diag(l_unsorted),'descend');
evs = evs_unsorted(:,idx);
minpos_l = max(l);
% replace redundant small negative values with minimal positive l, ...
% result of computational errors
for i = 1:samples
    if (l(i) > 0 && l(i) < minpos_l) 
        minpos_l = l(i);
    end
end
for i = 1:samples
    if(l(i) < minpos_l)
        l(i) = minpos_l;
    end
end

% mapping to basis vectors
bvs = (C./(sqrt(l)))'*evs;

% explained variance (lamda_i/sum(lamda))
l_expl = zeros(samples,1);
l_sum = sum(l);
for i = 1:samples
    l_expl(i) = sum(l(1:i))/l_sum;
end

% reduce to d dimensions
d = 2;
bvs_redu = bvs(:,1:d);
digits_projected = bvs_redu'*digits_train';



figure
gscatter(digits_projected(2,:), digits_projected(1,:), ...
    labels_train(flip(idx,1)))
figure
axes('LineWidth',0.6,...
    'FontName','Helvetica',...
    'FontSize',8,...
    'XAxisLocation','Origin',...
    'YAxisLocation','Origin')
xlim([1 300])
line(1:samples,l_expl);

%%
function [c] = center_datapoints(datapoints)
    c = datapoints - mean(datapoints);
end

