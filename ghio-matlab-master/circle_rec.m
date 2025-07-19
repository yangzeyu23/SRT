% circular_rec.m
% 功能: 从模拟的圆形光斑衍射图样中重建原始圆形光斑图像
% 算法: Guided Shrink-Wrap, GSW
% 依赖函数: ef.m, er.m, gshrinkwrap.m, hio2d.m, myalign.m

%% 1. 初始化
clc;        
clear all;  
close all;  
restoredefaultpath; 
addpath(genpath('.')); 

disp('开始圆形光斑衍射图像重建。');
disp('----------------------------------------------------');

%%  2. 生成原始圆形二值图像 (作为重建目标) 

image_size = 256; 
radius_ratio = 0.25; 
object = zeros(image_size, image_size, 'double'); 
center_x = image_size / 2 + 0.5;
center_y = image_size / 2 + 0.5;
circle_radius = image_size * radius_ratio;

for i = 1:image_size
    for j = 1:image_size
        distance = sqrt((i - center_y)^2 + (j - center_x)^2);
        if distance <= circle_radius
            object(i, j) = 1;
        end
    end
end

test_wavefront = object;  % 将原始图像作为 test_wavefront 用于后续误差计算
disp('原始圆形光斑（重建目标）已生成。');

%%  3. 模拟衍射：计算频率域分布强度 (作为 GSW 输入) 

F_object = fft2(object); 
Fabs_data = abs(F_object); 
Fabs_data_shifted = fftshift(Fabs_data);

noise_level = 0.00;    % 噪声水平 (可选)
Fabs_noisy = Fabs_data_shifted + noise_level * max(Fabs_data(:)) * randn(size(Fabs_data_shifted));

if any(isnan(Fabs_noisy(:))) || any(isinf(Fabs_noisy(:)))
    error('衍射图样数据 Fabs_noisy 包含 NaN 或 Inf 值，请检查数据生成过程。');
Fabs_noisy = Fabs_data_shifted; % 确保 Fabs_noisy 不为空或无效
end
disp('模拟衍射图样已计算完成，将作为重建算法的输入。');

%%  4. 定义 GSW 重建参数 

% [R, Sup, Rtmp, efs] = gshrinkwrap(Fabs, n1, checker, gen, n2, rep, alpha, sigma, cutoff1, cutoff2, beta)
config = struct(...
    'n1',       5000,      ... % 初始 HIO 迭代次数
    'gen',      300,       ... % GSW 世代数 (支持更新次数)
    'n2',       80,        ... % 每代 HIO 内部迭代次数
    'rep',      4,         ... % 并行副本数 (用于引导式收缩包裹的鲁棒性)
    'alpha',    30,        ... % OSS 平滑系数 (过采样平滑，用于 hio2d.m)
    'sigma',    1.5,       ... % 高斯模糊初始标准差 (用于生成新支撑)
    'cutoff1',  0.02,      ... % 自相关阈值 (用于从自相关图生成初始支撑)
    'cutoff2',  0.08,      ... % 支撑更新阈值 (用于从平滑图像生成新支撑)
    'beta',     0.85,      ... % HIO 反馈系数 (用于 hio2d.m)
    'checker',  false(image_size, image_size) ... % 默认没有需要忽略的频率域区域
);

% 定义一个用于跳过中心亮点的掩膜 (Checker mask)
checker_radius = 5;
[X, Y] = meshgrid(1:image_size, 1:image_size);
dist_from_center = sqrt((X - (image_size/2 + 0.5)).^2 + (Y - (image_size/2 + 0.5)).^2);
config.checker = dist_from_center > checker_radius; % true表示有效区域，false表示跳过

fprintf('GSW重建参数配置:\n');
fprintf('  总世代数: %d\n', config.gen);
fprintf('  每代 HIO 迭代: %d\n', config.n2);
fprintf('  并行副本数: %d\n', config.rep);
fprintf('  初始高斯核 Sigma: %.1f\n', config.sigma);
fprintf('  支撑更新阈值 (cutoff2): %.2f\n', config.cutoff2);
fprintf('  HIO 反馈系数 (beta): %.2f\n', config.beta);

%%  5. GSW 重建 

disp(' ');
disp('------ 开始执行 GSW 算法 ------');
try
    % 启动并行池 (如果未启动)
    if isempty(gcp('nocreate'))
        % 限制并行核心数，避免资源过度占用
        parpool('Processes', min(config.rep, feature('numcores')));
    end
    
    tic;
    [R_reconstructed_all_reps, S_all_gens, ~, efs_all_reps] = gshrinkwrap(...
        Fabs_noisy, ... % 衍射幅度数据 (含噪声)
        config.n1, ... % 初始迭代次数
        config.checker, ... % 未知区域标记 (这里没有)
        config.gen, ... % 收缩包裹代数
        config.n2, ... % 每代迭代次数
        config.rep, ... % 并行副本数
        config.alpha, ... % OSS参数
        config.sigma, ... % 高斯模糊初始标准差
        config.cutoff1, ... % 自相关阈值 (用于内部生成初始支撑)
        config.cutoff2, ... % 支撑更新阈值
        config.beta); % HIO反馈系数
    time_elapsed = toc; % 结束计时

    disp(' ');
    disp(['重建完成! 总耗时: ', num2str(time_elapsed/60, '%.1f'), ' 分钟']);

catch ME
    warning('重建失败！正在尝试关闭并行池，错误信息：%s', ME.message);
    % 确保错误发生时关闭并行池
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end
    rethrow(ME); % 重新抛出错误，中断脚本执行
end

% 确保关闭并行池
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

%%  6. 误差计算 
[~, best_rep_idx] = min(efs_all_reps(end,:)); % 找到最后一世代EF误差最小的副本索引
R_final_reconstructed = R_reconstructed_all_reps(:,:,best_rep_idx); % 获取最终重建结果

% 计算最终的EF和ER误差
final_ef = ef(Fabs_noisy, fft2(R_final_reconstructed), config.checker);
final_er = er(test_wavefront, abs(R_final_reconstructed), []);

fprintf('\n----------------------------------------------------\n');
fprintf('重建完成！\n');
fprintf('原始最佳副本的最终EF误差: %.4f\n', final_ef);
fprintf('原始最佳副本的最终ER误差: %.4f\n', final_er);

%% 7. 自动对齐 
fprintf('\n----------------------------------------------------\n');
fprintf('  自动对齐模式：正在识别重建图像中心并对齐。\n');
fprintf('----------------------------------------------------\n');

% 图像二值化以识别对象; 找到重建图像的非零部分，并进行归一化
reco_abs = abs(R_final_reconstructed);
% 设定一个阈值来二值化图像
binary_reco = imbinarize(reco_abs); 
% 标记连通区域并计算其属性
stats = regionprops(binary_reco, 'Area', 'Centroid');

if isempty(stats)
    warning('未在重建图像中找到任何连通区域，无法进行自动对齐。');
    R_reconstructed_auto_aligned = R_final_reconstructed; % 无法对齐，保持原样
    auto_ef_after_shift = final_ef;
    auto_er_after_shift = final_er;
else
    % 找到面积最大的连通区域，通常是目标对象
    all_areas = [stats.Area];
    [~, largest_idx] = max(all_areas);
    
    % 获取最大区域的质心
    center_of_object = stats(largest_idx).Centroid;
    
    % 计算图像中心点
    [rows, cols] = size(R_final_reconstructed);
    image_center_x = cols / 2 + 0.5;
    image_center_y = rows / 2 + 0.5;
    
    % 计算所需的平移量
    shift_x_auto = round(image_center_x - center_of_object(1));
    shift_y_auto = round(image_center_y - center_of_object(2));
    
    fprintf('检测到自动平移量: X=%d, Y=%d\n', shift_x_auto, shift_y_auto);
    
    % 应用自动平移
    R_reconstructed_auto_aligned = circshift(R_final_reconstructed, [shift_y_auto, shift_x_auto]);
    
    % 重新计算自动对齐后的EF和ER误差
    auto_ef_after_shift = ef(Fabs_noisy, fft2(R_reconstructed_auto_aligned), config.checker);
    auto_er_after_shift = er(test_wavefront, abs(R_reconstructed_auto_aligned), []);
    fprintf('自动对齐后 EF 误差: %.4f, ER 误差: %.4f\n', auto_ef_after_shift, auto_er_after_shift);
end

%% 8. 可视化

figure('Name', '重建结果对比', 'NumberTitle', 'off', 'Position', [50, 50, 1600, 800]); 

% 原始图像
subplot(2, 4, 1);
imshow(test_wavefront, []);
title('原始图像');
colormap(gca, gray);

% 最终收敛支撑
subplot(2, 4, 2);
if size(S_all_gens, 3) == config.gen + 1
    imshow(S_all_gens(:,:,config.gen+1), []); 
    title('最终收敛支撑');
else
    text(0.5, 0.5, '无法获取最终支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 衍射图样 (对数显示)
subplot(2, 4, 3);
imshow(log(Fabs_data_shifted + 1), []);
title('衍射图样 (对数显示)');
colormap(gca, jet);

% 收敛曲线 (各代最佳EF)
subplot(2, 4, 4);
if ~isempty(efs_all_reps) && size(efs_all_reps, 1) == config.gen + 1
    best_efs_per_gen = min(efs_all_reps, [], 2); % 每代所有副本中的最佳EF
    plot(0:config.gen, best_efs_per_gen, 'b', 'LineWidth', 2); % 从第0代开始
    xlabel('世代数');
    ylabel('最佳EF误差');
    title('重建收敛曲线');
    grid on;
else
    warning('efs_all_reps 的维度不符合预期，无法绘制完整的收敛曲线。');
    disp(['efs_all_reps 实际行数: ', num2str(size(efs_all_reps, 1))]);
    disp(['预期行数 (config.gen + 1): ', num2str(config.gen + 1)]);
    if ~isempty(efs_all_reps)
        best_efs_per_gen = min(efs_all_reps, [], 2);
        plot(0:size(best_efs_per_gen, 1)-1, best_efs_per_gen, 'b', 'LineWidth', 2);
        xlabel('世代数');
        ylabel('最佳EF误差');
        title('重建收敛曲线 (部分)');
        grid on;
    end
end

% 原始重建结果
subplot(2, 4, 5);
imshow(abs(R_final_reconstructed), []);
title(['原始重建 (EF=', num2str(final_ef, '%.3f'), ', ER=', num2str(final_er, '%.3f'), ')']);
colormap(gca, gray); 

% 原始绝对误差图
subplot(2, 4, 6);
error_map_original = abs(abs(R_final_reconstructed) - test_wavefront);
imshow(error_map_original, []);
title('原始绝对误差图');
colormap(gca, jet); 

% 自动对齐后的重建结果
subplot(2, 4, 7);
imshow(abs(R_reconstructed_auto_aligned), []);
title(['对齐后重建 (EF=', num2str(auto_ef_after_shift, '%.3f'), ', ER=', num2str(auto_er_after_shift, '%.3f'), ')']);
colormap(gca, gray); 
 

% 自动对齐后的绝对误差图 (现在更大)
subplot(2, 4, 8);
auto_error_map = abs(abs(R_reconstructed_auto_aligned) - test_wavefront);
imshow(auto_error_map, []);
title('对齐后绝对误差图');
colormap(gca, jet); 


final_reconstruction_image = abs(R_reconstructed_auto_aligned);
final_support_mask = S_all_gens(:,:,config.gen+1);  % 最终支撑 (保持不变)
final_ef_value = auto_ef_after_shift;
final_er_value = auto_er_after_shift;

disp(' ');
disp('脚本运行完成。所有重建结果和评估指标已显示在图窗中，并已保存到工作区变量。');