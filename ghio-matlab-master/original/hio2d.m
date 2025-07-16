% 2-D HIO written by Po-Nan Li @ Academia Sinica 2012
function R = hio2d(Fabs, S, n, varargin) % Fabs, S, n, unknown, alpha
    

    if isempty(varargin)
        unknown = false(size(Fabs));
    else
        unknown = varargin{1};
    end
    % OSS module
    if length(varargin) > 1 && ~isempty( varargin{2} )
        alpha = varargin{2};
        disp(['OSS is on. alpha = ' num2str(alpha)]);
        oss = true;
        x = -round((length(Fabs)-1)/2):round((length(Fabs)-1)/2);
        [X, Y] = meshgrid(x, x);
        W = exp(-0.5 .* (X./alpha).^2) .* exp(-0.5 .* (Y./alpha).^2);
        W = ifftshift(W);
    else
        oss = false;
    end
    % solve unknown pixels in data
    
    
    beta1 = 0.9;
    
    % generate random initial phases
    if sum(imag(Fabs(:))) == 0
        ph_init = rand(size(Fabs));
        ph_init = angle(fft2(ph_init));
        F = Fabs .* exp(1j.*ph_init);
    else
        F = Fabs;
    end
    
    F0 = abs(F); 
    previous = ifft2(F, 'symmetric');
    
    % ================ iterations ==================================
    for t = 1:n
        if mod(t-1, 100) == 0 && n >= 500
            disp(['step ' int2str(t)]);
        end
        rs = ifft2(F, 'symmetric'); % real space version
        cond1 = ~S | (rs<0);
        rs(cond1) = previous(cond1) - beta1 .* rs(cond1);
        previous = rs;
        if oss
            rs_oss = ifft2(fft2(rs) .* W, 'symmetric');
            rs(~S) = rs_oss(~S);
        end
        F2 = fft2(rs); % .* exp(-1j.*(U+V));
        F = F0 .* exp(1j.*angle(F2));
        F(unknown) = F2(unknown);
    end
        % ================ iterations ends here  ==================================
    R = ifft2(F, 'symmetric');
end