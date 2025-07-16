% circular_aperture_reconstruction.m

% =========================================================================
% 脚本名称: circular_aperture_reconstruction.m
% 功能: 从模拟的圆形光斑衍射图样中重建原始圆形光斑图像
% 算法: 使用引导式收缩包裹 (Guided Shrink-Wrap, GSW) 算法
% 依赖函数: ef.m, er.m, gshrinkwrap.m, hio2d.m, myalign.m
% (确保这些文件与本脚本在同一目录下)
% =========================================================================

%% ==================== 0. 初始化设置 ====================
clc;        % 清除命令行窗口
clear all;  % 清除工作区变量
close all;  % 关闭所有图形窗口
restoredefaultpath; % 恢复默认路径
addpath(genpath('.')); % 将当前文件夹及其所有子文件夹添加到 MATLAB 路径中

disp('----------------------------------------------------');
disp('   开始圆形光斑衍射图像重建：引导式收缩包裹算法');
disp('   (使用指定的原始函数集)');
disp('----------------------------------------------------');

%% ==================== 1. 定义图像和光斑参数 ====================
image_size = 256; % 仿真图像的像素尺寸 (N x N 像素)
radius_ratio = 0.25; % 圆形光斑半径占图像尺寸一半的比例

fprintf('原始圆形光斑生成参数:\n');
fprintf('  图像尺寸: %d x %d 像素\n', image_size, image_size);
fprintf('  光斑半径占图像一半尺寸的比例: %.2f\n', radius_ratio);

%% ==================== 2. 生成原始圆形二值图像 (作为重建目标) ====================
object = zeros(image_size, image_size, 'double'); % 创建全零矩阵，确保double类型

% 计算图像中心坐标 (MATLAB 索引从1开始)
center_x = image_size / 2 + 0.5;
center_y = image_size / 2 + 0.5;

% 计算圆形半径 (像素单位)
circle_radius = image_size * radius_ratio;

% 遍历像素，设置圆形区域为1
for i = 1:image_size
    for j = 1:image_size
        distance = sqrt((i - center_y)^2 + (j - center_x)^2);
        if distance <= circle_radius
            object(i, j) = 1; % 圆形区域设置为1
        end
    end
end

% 将原始图像作为 test_wavefront 用于后续误差计算
test_wavefront = object;

disp('原始圆形光斑（重建目标）已生成。');

%% ==================== 3. 模拟衍射：计算频率域分布强度 (作为 GSW 输入) ====================
F_object = fft2(object); % 对原始图像进行傅里叶变换
Fabs_data = abs(F_object); % 提取频率域分布强度 (幅度)

% 将零频率分量（中心）移动到中心，用于可视化
Fabs_data_shifted = fftshift(Fabs_data);

% 添加高斯噪声来模拟真实实验数据
noise_level = 0.01; % 1% 噪声水平 (可根据需要调整)
% Fabs_noisy 是 GSW 算法的实际输入，确保为 double 类型
Fabs_noisy = double(abs(Fabs_data + noise_level * max(Fabs_data(:)) * randn(size(Fabs_data))));

% 检查 Fabs_noisy 中是否存在 NaN 或 Inf
if any(isnan(Fabs_noisy(:))) || any(isinf(Fabs_noisy(:)))
    error('衍射图样数据 Fabs_noisy 包含 NaN 或 Inf 值，请检查数据生成过程。');
end

disp('模拟衍射图样（含噪声）已计算完成，将作为重建算法的输入。');

%% ==================== 4. 定义引导式收缩包裹重建参数 ====================
% 参数参考常见的 GSW 实现，并与 gshrinkwrap 函数的预期参数匹配
% 假设 gshrinkwrap.m 函数的签名为:
% [R, Sup, Rtmp, efs] = gshrinkwrap(Fabs, n1, checker, gen, n2, rep, alpha, sigma, cutoff1, cutoff2, beta)
config = struct(...
    'n1',       800,       ... % 初始 HIO 迭代次数
    'gen',      12,        ... % 收缩包裹代数 (支持更新次数)
    'n2',       300,       ... % 每代 HIO 内部迭代次数
    'rep',      4,         ... % 并行副本数 (用于引导式收缩包裹的鲁棒性)
    'alpha',    15,        ... % OSS 平滑系数 (过采样平滑，用于 hio2d.m)
    'sigma',    3.5,       ... % 高斯模糊初始标准差 (用于生成新支撑)
    'cutoff1',  0.05,      ... % 自相关阈值 (用于从自相关图生成初始支撑)
    'cutoff2',  0.12,      ... % 支撑更新阈值 (用于从平滑图像生成新支撑)
    'beta',     0.88,      ... % HIO 反馈系数 (用于 hio2d.m)
    'checker',  false(image_size, image_size) ... % 默认没有需要忽略的频率域区域
);

fprintf('引导式收缩包裹重建参数配置:\n');
fprintf('  总世代数: %d\n', config.gen);
fprintf('  每代 HIO 迭代: %d\n', config.n2);
fprintf('  并行副本数: %d\n', config.rep);
fprintf('  初始高斯核 Sigma: %.1f\n', config.sigma);
fprintf('  支撑更新阈值 (cutoff2): %.2f\n', config.cutoff2);
fprintf('  HIO 反馈系数 (beta): %.2f\n', config.beta);

%% ==================== 5. 执行引导式收缩包裹重建 ====================
disp(' ');
disp('===== 开始执行引导式收缩包裹重建算法 =====');

try
    % 启动并行池 (如果未启动)
    if isempty(gcp('nocreate'))
        % 限制并行核心数，避免资源过度占用
        parpool('Processes', min(config.rep, feature('numcores')));
    end
    
    tic; % 开始计时
    % 调用 gshrinkwrap 函数进行重建
    % 假设 gshrinkwrap.m 会内部处理初始支撑生成和多副本引导
    [R_reconstructed_all_reps, S_all_gens, ~, efs_all_reps] = gshrinkwrap(...
        Fabs_noisy, ...               % 衍射幅度数据 (含噪声)
        config.n1, ...                % 初始迭代次数
        config.checker, ...           % 未知区域标记 (这里没有)
        config.gen, ...               % 收缩包裹代数
        config.n2, ...                % 每代迭代次数
        config.rep, ...               % 并行副本数
        config.alpha, ...             % OSS参数
        config.sigma, ...             % 高斯模糊初始标准差
        config.cutoff1, ...           % 自相关阈值 (用于内部生成初始支撑)
        config.cutoff2, ...           % 支撑更新阈值
        config.beta);                 % HIO反馈系数
    time_elapsed = toc; % 结束计时
    
    disp(' ');
    disp(['重建完成! 总耗时: ', num2str(time_elapsed/60, '%.1f'), ' 分钟']);

catch ME
    % 捕获错误并显示，确保并行池关闭
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end
    % 重新抛出错误，并指明错误可能来源于 gshrinkwrap 或其依赖函数内部
    error('重建过程中在 gshrinkwrap 或其依赖函数内部发生错误: %s', ME.message);
end

%% ==================== 6. 结果评估 ====================
% 选择最佳重建结果 (最终代中EF最低的副本)
% efs_all_reps 是一个 (gen+1) x rep 的矩阵，表示每代每个副本的EF误差
[~, best_rep_idx] = min(efs_all_reps(end, :)); % 找到最终代EF最低的副本索引
R_final_reconstructed = R_reconstructed_all_reps(:,:,best_rep_idx); % 提取最佳重建结果

% 计算频率域误差 (EF)
% ef 函数期望幅度作为输入，而不是强度
final_ef_error = ef(Fabs_noisy, fft2(R_final_reconstructed), config.checker);
disp(['最终频率域误差 (EF): ', num2str(final_ef_error, '%.4f')]);

% 计算实空间误差 (ER)
% er 函数期望原始图像和重建图像幅度作为输入
final_er_error = er(test_wavefront, abs(R_final_reconstructed), []);
disp(['最终实空间误差 (ER): ', num2str(final_er_error, '%.4f')]);

%% ==================== 7. 可视化结果 ====================
figure('Name', '圆形光斑重建结果', 'Position', [100, 100, 1400, 600]);

% 原始图像 (重建目标)
subplot(2, 4, 1);
imshow(test_wavefront, []);
title('原始圆形光斑');
colormap(gca, gray);

% 模拟的衍射图样 (FFT 幅度对数显示)
subplot(2, 4, 2);
imshow(log(Fabs_data_shifted + 1), []); % 加1避免log(0)
title('模拟衍射图样 (对数显示)');
colormap(gca, hot); % 通常用于频率域热图

% 重建结果 (最终副本)
subplot(2, 4, 3);
imshow(abs(R_final_reconstructed), []); % 显示重建结果的幅度
title(['重建结果 (EF=', num2str(final_ef_error, '%.3f'), ')']);
colormap(gca, gray);

% 绝对误差图
subplot(2, 4, 4);
error_map = abs(abs(R_final_reconstructed) - test_wavefront);
imshow(error_map, []);
title('绝对误差图');
colormap(gca, jet); % 使用jet色图便于观察误差分布
colorbar;

% 初始支撑 (从 gshrinkwrap 内部获取，或者我们可以基于外部定义的cutoff1重绘)
% 为了与 gshrinkwrap 内部生成方式一致，这里不再直接使用 S_initial，
% 而是尝试从 S_all_gens 中提取第1代支撑。
subplot(2, 4, 5);
if ~isempty(S_all_gens) && size(S_all_gens, 3) >= 1
    imshow(S_all_gens(:,:,1), []);
    title('初始支撑 (gshrinkwrap内部生成)');
else
    text(0.5, 0.5, '无法获取初始支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 最终收敛的支撑
subplot(2, 4, 6);
if ~isempty(S_all_gens) && size(S_all_gens, 3) == config.gen + 1
    imshow(S_all_gens(:,:,config.gen+1), []); 
    title('最终收敛支撑');
else
    text(0.5, 0.5, '无法获取最终支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 收敛曲线 (各代最佳EF)
subplot(2, 4, 7);
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
        plot(0:size(best_efs_per_gen, 1)-1, best_efs_per_gen, 'r--', 'LineWidth', 2);
        xlabel('世代数');
        ylabel('EF误差');
        title('重建收敛曲线 (部分)');
        grid on;
    else
        text(0.5, 0.5, '无法绘制收敛曲线', 'HorizontalAlignment', 'center');
    end
end

% 3D可视化重建结果 (可选，可能需要较长时间渲染)
subplot(2, 4, 8);
surf(abs(R_final_reconstructed), 'EdgeColor', 'none');
view(-30, 60);
title('重建结果 3D 视图');
colormap(jet);
light;
lighting gouraud;
axis tight;

disp('----------------------------------------------------');
disp('   脚本执行完毕。请检查生成的图形窗口。');
disp('----------------------------------------------------');

% 提示关闭并行池
if ~isempty(gcp('nocreate'))
    fprintf('\n若不再需要并行计算，请在命令行输入: delete(gcp)\n');
end