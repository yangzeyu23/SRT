function R = hio2d(Fabs, S, n, varargin)
    % 2-D HIO with enhanced size compatibility
    % Input validation
    if ~ismatrix(Fabs) || ~ismatrix(S)
        error('Inputs must be 2D matrices');
    end
    if ~isequal(size(Fabs), size(S))
        error('Fabs and S must have same dimensions');
    end

    [rows, cols] = size(Fabs);
    
    % Handle optional arguments
    if isempty(varargin)
        unknown = false(size(Fabs));
    else
        unknown = varargin{1};
    end

    % Enhanced OSS module
    if length(varargin) > 1 && ~isempty(varargin{2})
        alpha = varargin{2};
        oss = true;
        
        % Size-adaptive Gaussian filter
        x = (1:cols) - ceil(cols/2);
        y = (1:rows) - ceil(rows/2);
        [X, Y] = meshgrid(x, y);
        W = exp(-0.5*(X/alpha).^2) .* exp(-0.5*(Y/alpha).^2);
        W = ifftshift(W);
    else
        oss = false;
    end

    % Initialization
    beta1 = 0.9;
    if sum(abs(imag(Fabs(:)))) == 0
        ph_init = rand(size(Fabs));
        ph_init = angle(fft2(ph_init));
        F = Fabs .* exp(1j*ph_init);
    else
        F = Fabs;
    end

    F0 = abs(F);
    previous = ifft2(F, 'symmetric');

    % Main iteration loop
    for t = 1:n
        if mod(t-1, 100) == 0 && n >= 500
            disp(['step ' int2str(t)]);
        end

        rs = ifft2(F, 'symmetric');
        cond1 = ~S | (rs<0);
        rs(cond1) = previous(cond1) - beta1 * rs(cond1);
        previous = rs;

        % Size-checked OSS application
        if oss
            rs_oss = ifft2(fft2(rs) .* W, 'symmetric');
            if ~isequal(size(rs_oss), size(rs))
                error('OSS filter size mismatch');
            end
            rs(~S) = rs_oss(~S);
        end

        F2 = fft2(rs);
        F = F0 .* exp(1j*angle(F2));
        F(unknown) = F2(unknown);
    end
    
    R = ifft2(F, 'symmetric');
end
