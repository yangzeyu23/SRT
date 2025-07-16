% [R, Sup, M] = gshrinkwrap(Fabs, n, checker, gen, n2, rep, varargin) 
% varargin = {alpha, sigma, cutoff1, cutoff2}
% GUIDED shrink-wrap 2-D HIO written by Po-Nan Li @ Academia Sinica 2014
% reference: 
% [1] Marchesini et al., 
%     "X-ray image reconstruction from a diffraction pattern alone,"
%     Phys. Rev. B 68, 140101 (2003).
% [2] Chen et al., Phys. Rev. B 76, 064113 (2007).
%     
% v.1 2014/06/03 : multiple seed
% v.2 2014/06/06 : multiple following runs
% v.3 2014/06/09 : use "template" to update each replica

function [R, Sup, Rtmp, efs] = gshrinkwrap(Fabs, n1, checker, gen, n2, rep, varargin)
    % Enhanced guided shrink-wrap with parallel support
    
    % Parameter handling
    alpha = [];
    sig = 3;
    cutoff1 = 0.04;
    cutoff2 = 0.2;
    
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

    % Initial support from autocorrelation
    S = fftshift(ifft2(abs(Fabs).^2, 'symmetric'));
    S = S > cutoff1*max(S(:));

    % Pre-allocation
    base_size = size(Fabs);
    R = zeros([base_size, gen+1]);
    Sup = false([base_size, gen+1]);
    Sup(:,:,1) = S;
    
    Rtmp = zeros([base_size, rep]);
    Ftmp = zeros([base_size, rep]);
    Mtmp = zeros([base_size, rep]);
    Stmp = false([base_size, rep]);
    efs = zeros(gen+1, rep);

    % Gaussian kernel
    x = (1:base_size(2)) - ceil(base_size(2)/2) - 1;
    y = (1:base_size(1)) - ceil(base_size(1)/2) - 1;
    [X, Y] = meshgrid(x, y);
    rad = sqrt(X.^2 + Y.^2);

    % First generation
    parfor r = 1:rep
        try
            Rtmp(:,:,r) = hio2d(Fabs, S, n1, checker, alpha);
            Ftmp(:,:,r) = fft2(Rtmp(:,:,r));
            efs(1,r) = ef(Fabs, Ftmp(:,:,r), checker);
        catch
            Rtmp(:,:,r) = zeros(base_size);
            efs(1,r) = inf;
        end
    end

    % Select best replica
    [~, mx] = min(efs(1,:));
    R(:,:,1) = Rtmp(:,:,mx);

    % Shrink-wrap iterations
    for g = 2:(gen+1)
        Rmodel = R(:,:,g-1);
        
        parfor r = 1:rep
            try
                % Alignment and mixing
                aligned = myalign(Rmodel, Rtmp(:,:,r));
                Rtmp(:,:,r) = sign(aligned) .* sqrt(abs(aligned .* Rmodel));
                
                % Support update
                G = fft2(exp(-(rad/sqrt(2)/sig).^2));
                Mtmp(:,:,r) = fftshift(ifft2(fft2(Rtmp(:,:,r)) .* G, 'symmetric'));
                Stmp(:,:,r) = (Mtmp(:,:,r) >= cutoff2*max(Mtmp(:,:,r), [], 'all'));
                
                % HIO with new support
                Rtmp(:,:,r) = hio2d(fft2(Rtmp(:,:,r)), Stmp(:,:,r), n2, checker, alpha);
                Ftmp(:,:,r) = fft2(Rtmp(:,:,r));
                efs(g,r) = ef(Fabs, Ftmp(:,:,r), checker);
            catch
                efs(g,r) = inf;
            end
        end
        
        % Selection
        [~, mx] = min(efs(g,:));
        R(:,:,g) = Rtmp(:,:,mx);
        Sup(:,:,g) = Stmp(:,:,mx);
        
        % Kernel shrinkage
        if sig > 1.5
            sig = sig * 0.99;
        end
    end
end