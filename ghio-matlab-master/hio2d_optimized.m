function R = hio2d_optimized(Fabs, S, n, unknown, alpha, beta)
%% 输入验证
if nargin < 5
    alpha = [];  % 默认关闭OSS
end
if nargin < 6
    beta = 0.9;  % 默认反馈系数
end

validateattributes(Fabs, {'single','double'}, ...
    {'2d', 'nonnegative', 'real'}); % 强制非负实数输入

if any(Fabs(:) < 0)
    error('输入Fabs包含负值, 请检查数据预处理流程');
end

%% 初始化
[rows, cols] = size(Fabs);
if isempty(unknown)
    unknown = false(rows, cols);
end

% 随机相位初始化
if isreal(Fabs)
    phase = angle(fft2(rand(rows, cols)));
    F = Fabs .* exp(1j*phase);
else
    F = Fabs;
end

F0 = abs(F);
previous = ifft2(F, 'symmetric');

%% OSS滤波器生成
if ~isempty(alpha)
    [x, y] = meshgrid(...
        (1:cols) - ceil(cols/2), ...
        (1:rows) - ceil(rows/2));
    W = exp(-((x/alpha).^2 + (y/alpha).^2)/2);
    W = ifftshift(W);
else
    W = [];
end

%% 主迭代循环
for t = 1:n
    % 实时进度显示
    if mod(t,100) == 0
        fprintf('迭代 %d/%d (%.1f%%)\n', t, n, t/n*100);
    end
    
    % 实空间更新
    rs = ifft2(F, 'symmetric');
    cond = ~S | (rs < 0);
    rs(cond) = previous(cond) - beta * rs(cond);
    
    % OSS平滑
    if ~isempty(W)
        rs_oss = ifft2(fft2(rs) .* W, 'symmetric');
        rs(~S) = rs_oss(~S);
    end
    
    % 频域更新
    F2 = fft2(rs);
    F = F0 .* exp(1j*angle(F2));
    F(unknown) = F2(unknown);
    
    previous = rs;  % 保存当前状态
end

R = ifft2(F, 'symmetric');
end