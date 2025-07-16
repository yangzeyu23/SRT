function grayscale_optimized(image_path)

%% ==================== 初始化设置 ====================
clc; close all; 
restoredefaultpath;
addpath(genpath('.'));

%% ==================== 第一阶段：数据安全加载 ====================
if ~exist('image_path', 'var') || isempty(image_path)
    [file, path] = uigetfile({'*.png;*.jpg;*.tif;*.dcm', '图像文件'});
    if isequal(file, 0)
        error('用户取消选择');
    end
    image_path = fullfile(path, file);
end

try
    % 安全读取图像
    img_info = imfinfo(image_path);
    if ismember(img_info.ColorType, {'grayscale', 'indexed'})
        img_raw = imread(image_path);
    else
        img_raw = rgb2gray(imread(image_path));
    end
    
    % 强制非负化
    img_processed = single(abs(img_raw));
    disp('图像加载完成，已强制非负化');
catch ME
    error('图像加载失败: %s', ME.message);
end

%% ==================== 第二阶段：图像预处理 ====================
target_size = 256;  % 提高分辨率
bit_depth = 32;     % 增加灰度级数

% 安全的归一化处理
img_normalized = mat2gray(img_processed); % 自动缩放至[0,1]
test_wavefront = imresize(img_normalized, [target_size, target_size]);

% 验证数据范围
if any(test_wavefront(:) < 0)
    warning('发现负像素值，自动取绝对值');
    test_wavefront = abs(test_wavefront);
end

%% ==================== 第三阶段：衍射模拟 ====================
F_test = fft2(test_wavefront);
Fabs = abs(F_test);
Fabs_shifted = fftshift(Fabs);


noise_level = 0.00;
Fabs_noisy = abs(Fabs + noise_level*max(Fabs(:))*randn(size(Fabs))); % 强制非负
disp('衍射数据已添加噪声并验证非负性');

% 验证频域数据非负
if any(Fabs_noisy(:) < 0)
    Fabs_noisy = abs(Fabs_noisy);
    warning('频域数据含负值，已自动校正');
end

%% ==================== 第四阶段：重建参数配置 ====================
config = struct(...
    'n1',       1000,       ... % 初始HIO迭代（原1000→减少20%）
    'gen',      30,        ... % 收缩包裹代数（原15→减少3代）
    'n2',       500,       ... % 每代HIO迭代（原500→减少40%）
    'rep',      16,        ... % 保持16个并行副本
    'alpha',    20,        ... % OSS平滑系数（原18→稍减弱平滑）
    'sigma',    4,       ... % 初始高斯核（原4.0→加速收缩）
    'cutoff1',  0.04,      ... % 自相关阈值（原0.03→更严格初始支撑）
    'cutoff2',  0.2,      ... % 支撑阈值（原0.15→更敏感更新）
    'beta',     0.9,      ... % HIO反馈系数（原0.85→增强收敛）
    'checker',  false(target_size) ...
);

%% 调用前验证

disp('=== 调用gshrinkwrap_optimized前验证 ===');
disp(['Fabs_noisy大小: ', mat2str(size(Fabs_noisy))]);
disp(['数据类型：', class(Fabs_noisy)]);
disp(['数值范围：', num2str([min(Fabs_noisy(:)), max(Fabs_noisy(:))])]);

%% ==================== 第五阶段：并行重建 ====================
try
    % 启动并行池
    if isempty(gcp('nocreate'))
        parpool('Processes', min(config.rep, feature('numcores')));
    end
    
    tic;
    [R, Sup, ~, efs] = gshrinkwrap_optimized(...
        Fabs_noisy, ...
        config.n1, ...
        config.checker, ...
        config.gen, ...
        config.n2, ...
        config.rep, ...
        config.alpha, ...
        config.sigma, ...
        config.cutoff1, ...
        config.cutoff2, ...
        config.beta);
    time_elapsed = toc;
    
    disp(['重建完成! 耗时: ', num2str(time_elapsed/60, '%.1f'), ' 分钟']);
catch ME
    delete(gcp('nocreate'));
    error('重建失败: %s', ME.message);
end

%% ==================== 第六阶段：结果评估 ====================
[~, best_rep] = min(efs(end,:));
R_best = R(:,:,best_rep);

% 计算评估指标
final_ef = ef(Fabs_noisy, fft2(R_best), config.checker);
final_er = er(test_wavefront, abs(R_best), []);

%% ==================== 第七阶段：可视化输出 ====================
figure('Name', '重建结果对比', 'Position', [100,100,1200,400]);
subplot(1,3,1); imshow(test_wavefront, []); title('原始图像');
subplot(1,3,2); imshow(log(Fabs_shifted+1), []); title('衍射图案');
subplot(1,3,3); imshow(abs(R_best), []); 
title(['重建结果 (EF=', num2str(final_ef, '%.3f'), ')']);

figure('Name', '支撑域演化');
montage(Sup, 'Size', [4,3], 'DisplayRange', []);
title('支撑域演化过程');

figure('Name', '收敛分析');
plot(min(efs,[],2), 'LineWidth', 2);
xlabel('代数'); ylabel('最佳EF值'); grid on;
title('收敛曲线');
end
