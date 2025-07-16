function [R, Sup, Rtmp, efs] = gshrinkwrap_optimized(Fabs, n1, checker, gen, n2, rep, alpha, sigma, cutoff1, cutoff2, beta)
%% 参数验证
validateattributes(Fabs, {'single','double'}, {'2d', 'nonnegative'});
if sigma <= 1 || sigma > 10
    error('sigma应在1-10之间');
end

%% 初始化设置
[rows, cols] = size(Fabs);
base_size = [rows, cols];

% 调试信息（修正：使用正确的输入参数名Fabs）
disp(['输入数据大小：', mat2str(size(Fabs))]);
disp(['最大值：', num2str(max(Fabs(:)))]);

% 生成坐标网格用于高斯核
[x, y] = meshgrid(...
    (1:cols) - floor(cols/2) - 1, ...
    (1:rows) - floor(rows/2) - 1);
rad = sqrt(x.^2 + y.^2);

%% 初始支撑生成
% 添加高斯平滑到自相关图，以提高初始支撑的鲁棒性
gaussian_kernel_for_autocorr = fspecial('gaussian', [7 7], 2); % 可以尝试不同的核大小和sigma值
autocorr_smooth = imfilter(autocorr, gaussian_kernel_for_autocorr, 'replicate');
S_initial = autocorr_smooth > max(config.cutoff1 * max(autocorr_smooth(:)), eps); % 注意这里要用autocorr_smooth

% 预分配内存
R = zeros([base_size, gen+1], 'single');
Sup = false([base_size, gen+1]);
Sup(:,:,1) = S_initial;

Rtmp = zeros([base_size, rep], 'single');
efs = inf(gen+1, rep);

%% 第一代：独立初始化
disp('-- 第一代初始化 --');
parfor r = 1:rep
    try
        % 使用不同随机种子
        rng(r);
        Rtmp(:,:,r) = hio2d(Fabs, S_initial, n1, checker, alpha, beta);
        efs(1,r) = ef(Fabs, fft2(Rtmp(:,:,r)), checker);
    catch ME
        warning('副本%d失败: %s', r, ME.message);
    end
end

% 选择最佳副本
[~, best] = min(efs(1,:));
R(:,:,1) = Rtmp(:,:,best);
disp(['最佳初始EF: ', num2str(efs(1,best)), ' (副本', num2str(best), ')']);

%% 主循环：引导式收缩包裹
for g = 2:gen+1
    fprintf('\n-- 第%d代 (共%d代) --\n', g-1, gen);
    R_guide = R(:,:,g-1);  % 引导模板
    
    % 预分配支撑域存储（修复：使用正确变量名）
    Stmp = false([base_size, rep]);
    
    parfor r = 1:rep
        try
            %% 步骤1：对齐与混合
            aligned = myalign(R_guide, Rtmp(:,:,r));
            mixed = sqrt(abs(aligned .* R_guide));  % 几何平均
            
            %% 步骤2：动态支撑更新
            current_sigma = max(sigma * 0.98^(g-2), 1.2);
            G = exp(-(rad/(sqrt(2)*current_sigma)).^2);
            M = fftshift(ifft2(fft2(mixed) .* fft2(G)));
            S_new = M > cutoff2 * max(M(:));
            
            %% 步骤3：约束重建
            Rtmp(:,:,r) = hio2d(fft2(mixed), S_new, n2, checker, alpha, beta);
            Stmp(:,:,r) = S_new; % 存储支撑域
            efs(g,r) = ef(Fabs, fft2(Rtmp(:,:,r)), checker);
        catch
            efs(g,r) = inf;
        end
    end
    
    % 选择最佳副本
    [min_ef, best] = min(efs(g,:));
    R(:,:,g) = Rtmp(:,:,best);
    Sup(:,:,g) = Stmp(:,:,best);
    
    fprintf('代数%d: 最佳EF=%.4f (副本%d)\n', g-1, min_ef, best);
end
end