% [R, Sup, M] = shrinkwrap(Fabs, n, checker, gen, n2, varargin) 
% varargin = {alpha, sigma, cutoff1, cutoff2}

function [R, Sup, M] = shrinkwrap(Fabs, n, checker, gen, n2, varargin) 

% default parameters;
alpha = [];
sig = 3;
cutoff1 = 0.04;
cutoff2 = 0.2;

% handle additional aruguments
if ~isempty(varargin)
    alpha = varargin{1};
    if length(varargin) > 1
        sig = varargin{2};
        if length(varargin) > 2
            cutoff1 = varargin{3};
            if length(varargin) > 3
                cutoff2 = varargin{4};
            end
        end
    end
end

S0 = double(fftshift( ifft2(abs(Fabs).^2, 'symmetric') ));  
S1 = S0 > cutoff1 * double(max(S0(:))); 


% pre-allocated spaces
R   = zeros( size(Fabs, 1), size(Fabs, 2), gen+1);
Sup = false( size(Fabs, 1), size(Fabs, 2), gen+1);
Sup(:,:,1) = S1;

% first run with initial support from auto-correlation map
R(:,:,1) = hio2d(Fabs, S1, n, checker, alpha);

% make Gaussian kernel
x = (1:size(S1,2)) - size(S1,2)/2;
y = (1:size(S1,1)) - size(S1,1)/2;
[X, Y] = meshgrid(x, y);
rad = sqrt(X.^2 + Y.^2);

% shrink-wrap
for g = 1:gen
    G = exp(-(rad./sqrt(2)./sig).^2);
    M = fftshift( ifft2( fft2(R(:,:,g)) .* fft2(G), 'symmetric') );
    Sup(:,:,g+1) = ( M >= cutoff2*max(M(:)) );
    R(:,:,g+1) = hio2d(fft2(R(:,:,g)), Sup(:,:,g+1), n2, checker, alpha);
    if sig > 1.5
        sig = sig * 0.99;
    end
end
    