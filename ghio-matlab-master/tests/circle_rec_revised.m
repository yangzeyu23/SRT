% circle_rec_revised.m
% 功能: 从模拟的圆形光斑衍射图样中重建原始圆形光斑图像
% 算法: Guided Shrink-Wrap (GSW)
% 依赖函数: ef.m, er.m, gshrinkwrap.m, hio2d.m, myalign.m, findpeaks2.m
% (注意: hiosupport.m 已移除，因 gshrinkwrap.m 内部自动生成初始支撑)

%% 1. 初始化
clc;        
clear all;  
close all;  
restoredefaultpath; 
addpath(genpath('.')); 

disp('开始圆形光斑衍射图像重建 (GSW)。');
disp('----------------------------------------------------');

%%  2. 生成原始圆形二值图像 (作为重建目标) 

original_image_size = 256; % 原始图像尺寸
radius_ratio = 0.25; 
object = zeros(original_image_size, original_image_size, 'double'); 
center_x = original_image_size / 2 + 0.5;
center_y = original_image_size / 2 + 0.5;
circle_radius = original_image_size * radius_ratio;

for i = 1:original_image_size
    for j = 1:original_image_size
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
end
disp('模拟衍射图样已计算完成，将作为重建算法的输入。');

% --- 折中解决方案核心：填充 Fabs_noisy 以规避 hio2d.m 中的 bug ---
% hio2d.m 中的 OSS 模块在处理偶数尺寸输入时存在bug。
% 通过将 Fabs_noisy 填充到一个奇数尺寸 (例如 257x257)，
% hio2d.m 将接收一个奇数尺寸的输入，从而使其 OSS 模块正常工作。
% 这是在不修改 hio2d.m 函数文件的前提下，激活 OSS 的折中方案。
padded_image_size = original_image_size + 1; % 例如，从 256 变为 257
Fabs_noisy_processed = padarray(Fabs_noisy, [1 1], 0, 'post'); % 在右侧和底部填充一行/列零

fprintf('傅里叶数据已从 %dx%d 填充至 %dx%d 以规避 hio2d.m 的 bug。\n', ...
    original_image_size, original_image_size, padded_image_size, padded_image_size);

%%  4. 定义 GSW 重建参数 

% [R, Sup, Rtmp, efs] = gshrinkwrap(Fabs, n, checker, gen, n2, rep, alpha, sigma, cutoff1, cutoff2)
config = struct(...
    'n1',       5000,      ... % 初始 HIO 迭代次数
    'gen',      300,       ... % GSW 世代数 (支持更新次数)
    'n2',       80,        ... % 每代 HIO 内部迭代次数
    'rep',      4,         ... % 并行副本数 (用于引导式收缩包裹的鲁棒性)
    'alpha',    30,        ... % OSS参数 (保持激活)
    'sigma',    1.5,       ... % 高斯模糊初始标准差 (用于生成新支撑)
    'cutoff1',  0.04,      ... % 自相关阈值 (用于从自相关图生成初始支撑，gshrinkwrap 默认值)
    'cutoff2',  0.2,       ... % 支撑更新阈值 (用于从平滑图像生成新支撑，gshrinkwrap 默认值)
    'beta',     0.85,      ... % HIO 反馈系数 (HIO 参数，虽然在gshrinkwrap内部，但如果hio2d支持可调整)
    'checker',  false(padded_image_size, padded_image_size) ... % Checker mask 也要匹配填充后的尺寸
);

% 定义一个用于跳过中心亮点的掩膜 (Checker mask)
% 在傅里叶空间，中心区域通常亮度最高，可能包含饱和或噪声，排除有助于稳定相位恢复。
checker_radius = 5;
[X, Y] = meshgrid(1:padded_image_size, 1:padded_image_size); % 匹配填充后的尺寸
dist_from_center = sqrt((X - (padded_image_size/2 + 0.5)).^2 + (Y - (padded_image_size/2 + 0.5)).^2);
config.checker = dist_from_center > checker_radius; 

fprintf('GSW重建参数配置:\n');
fprintf('  总世代数: %d\n', config.gen);
fprintf('  每代 HIO 迭代: %d\n', config.n2);
fprintf('  并行副本数: %d\n', config.rep);
fprintf('  初始高斯核 Sigma: %.1f\n', config.sigma);
fprintf('  支撑更新阈值 (cutoff2): %.2f\n', config.cutoff2);
fprintf('  OSS (alpha) 状态: 已激活 (%.1f)\n', config.alpha);

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
    [R_reconstructed_all_reps_padded, S_all_gens_padded, ~, efs_all_reps] = gshrinkwrap(...
        Fabs_noisy_processed, ... % 衍射幅度数据 (填充后的)
        config.n1, ... % 初始迭代次数 (用于第一代副本的 HIO 运行)
        config.checker, ... % 未知区域标记 (中心亮点)
        config.gen, ... % 收缩包裹代数 (支持更新的次数)
        config.n2, ... % 每代内部 HIO 迭代次数
        config.rep, ... % 并行副本数
        config.alpha, ... % OSS参数
        config.sigma, ... % 高斯模糊初始标准差
        config.cutoff1, ... % 自相关阈值 (用于内部生成初始支撑)
        config.cutoff2); % 支撑更新阈值

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

% --- 折中解决方案核心：裁剪重建结果回原始尺寸 ---
R_reconstructed_all_reps = R_reconstructed_all_reps_padded(1:original_image_size, 1:original_image_size, :);
S_all_gens = S_all_gens_padded(1:original_image_size, 1:original_image_size, :);

%%  6. 误差计算 
[~, best_rep_idx] = min(efs_all_reps(end,:)); % 找到最后一世代EF误差最小的副本索引
R_final_reconstructed = R_reconstructed_all_reps(:,:,best_rep_idx); % 获取最终重建结果

% 计算最终的EF和ER误差
% 注意：这里的 checker 和 Fabs_noisy 应该使用原始尺寸的版本进行误差计算
% 所以我们需要重新生成一个 original_sized_checker
original_checker = false(original_image_size, original_image_size);
[X_orig, Y_orig] = meshgrid(1:original_image_size, 1:original_image_size);
dist_from_center_orig = sqrt((X_orig - (original_image_size/2 + 0.5)).^2 + (Y_orig - (original_image_size/2 + 0.5)).^2);
original_checker = dist_from_center_orig > checker_radius; % 保持 checker_radius 不变

final_ef = ef(Fabs_noisy, fft2(R_final_reconstructed), original_checker); 
final_er = er(test_wavefront, abs(R_final_reconstructed), []); 

fprintf('\n----------------------------------------------------\n');
fprintf('重建完成！\n');
fprintf('原始最佳副本的最终EF误差: %.4f\n', final_ef);
fprintf('原始最佳副本的最终ER误差: %.4f\n', final_er);

%% 7. 图像对齐说明 (依赖 gshrinkwrap.m 内部的 myalign.m)
% gshrinkwrap.m 内部已经使用了 myalign.m 进行迭代间的对齐和融合。
% 因此，重建图像在算法内部已经得到有效对齐，无需在此处进行额外的质心校正。
fprintf('\n----------------------------------------------------\n');
fprintf('  重建图像在 GSW 算法内部已通过互相关方法自动对齐。\n');
fprintf('  （`myalign.m` 在 `gshrinkwrap.m` 内部被调用，确保副本间的良好融合）\n');
fprintf('----------------------------------------------------\n');

% 最终显示结果将是 gshrinkwrap 算法选择的最佳副本，它已经过内部对齐。
R_reconstructed_aligned_for_display = R_final_reconstructed; 

%% 8. 可视化

figure('Name', '重建结果对比', 'NumberTitle', 'off', 'Position', [50, 50, 1600, 800]); 

% 原始图像
subplot(2, 4, 1); % 调整子图布局为 2x4
imshow(test_wavefront, []);
title('原始图像');
colormap(gca, gray);

% 衍射图样 (对数显示)
subplot(2, 4, 2); % 调整子图布局
imshow(log(Fabs_data_shifted + 1), []);
title('衍射图样 (对数显示)');
colormap(gca, jet);

% 初始支撑 (gshrinkwrap 自动生成)
subplot(2, 4, 3); % 新增子图
% S_all_gens(:,:,1) 是 gshrinkwrap 内部使用的初始支撑，它也经过了填充
% 在此处将其裁剪回原始尺寸进行显示
imshow(S_all_gens(1:original_image_size, 1:original_image_size, 1), []); 
title('初始支撑 (gshrinkwrap 自动生成)');
colormap(gca, gray);

% 收敛曲线 (各代最佳EF)
subplot(2, 4, 4); % 调整子图布局
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

% 最终重建结果 (已通过 gshrinkwrap 内部对齐)
subplot(2, 4, 5); % 调整子图布局
imshow(abs(R_reconstructed_aligned_for_display), []);
title(['最终重建 (EF=', num2str(final_ef, '%.3f'), ', ER=', num2str(final_er, '%.3f'), ')']);
colormap(gca, gray); 
 
% 最终绝对误差图 (与原始图像对比)
subplot(2, 4, 6); % 调整子图布局
final_error_map = abs(abs(R_reconstructed_aligned_for_display) - test_wavefront);
imshow(final_error_map, []);
title('最终绝对误差图');
colormap(gca, jet); 

% 最终收敛支撑
subplot(2, 4, 7); % 调整子图布局
if size(S_all_gens, 3) == config.gen + 1
    imshow(S_all_gens(:,:,config.gen+1), []); 
    title('最终收敛支撑');
else
    text(0.5, 0.5, '无法获取最终支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 频率域中心掩膜 (原始尺寸) - 第8个位置，保留为空或用于未来扩展
subplot(2, 4, 8);
imshow(original_checker, []);
title('频率域中心掩膜 (原始尺寸)');
colormap(gca, gray);


final_reconstruction_image = abs(R_reconstructed_aligned_for_display);
final_support_mask = S_all_gens(:,:,config.gen+1);  % 最终支撑
final_ef_value = final_ef;
final_er_value = final_er;

% Helper function for conditional display
function result = ifelse(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

disp(' ');
disp('脚本运行完成。所有重建结果和评估指标已显示在图窗中，并已保存到工作区变量。');