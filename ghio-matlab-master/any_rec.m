% reconstruct_arbitrary_image.m
% 功能: 从任意输入的本地图片重建原始图像
% 算法: Guided Shrink-Wrap, GSW
% 依赖函数: ef.m, er.m, gshrinkwrap.m, hio2d.m, myalign.m, findpeaks2.m, hiosupport.m

%% 1. 初始化
clc;
clear all;
close all;
restoredefaultpath;
addpath(genpath('.'));

disp('开始任意图像重建。');
disp('----------------------------------------------------');

%% 2. 选择并预处理输入图像
try
    % 选择本地图片文件
    disp('请选择一个本地图片文件...');
    file_path = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.gif', '所有图像文件 (*.jpg;*.jpeg;*.png;*.bmp;*.gif)';
                           '*.*', '所有文件 (*.*)'}, '选择输入图像');

    if file_path == 0 
        disp('用户取消了文件选择。');
        return;
    end

    input_image_color = imread(file_path);
    disp(['已选择文件: ', file_path]);

    % 转换为灰度图
    if size(input_image_color, 3) == 3
        input_image_gray = rgb2gray(input_image_color);
    else
        input_image_gray = input_image_color;
    end

    % 裁剪为正方形 (如果需要)
    rows = size(input_image_gray, 1);
    cols = size(input_image_gray, 2);
    if rows ~= cols
        disp('输入图像不是正方形，正在进行中心裁剪...');
        min_dim = min(rows, cols);
        start_row = round((rows - min_dim) / 2) + 1;
        start_col = round((cols - min_dim) / 2) + 1;
        input_image_gray_cropped = input_image_gray(start_row:start_row + min_dim - 1, start_col:start_col + min_dim - 1);
        input_image_resized = imresize(input_image_gray_cropped, [256 256]);
    else
        input_image_resized = imresize(input_image_gray, [256 256]);
    end

    input_image_gray_8bit = uint8(255 * mat2gray(input_image_resized)); % 转换为 8-bit 灰度图

    % 保存预处理后的灰度图像
    output_gray_path = 'preprocessed_gray_image.png';
    imwrite(input_image_gray_8bit, output_gray_path);
    disp(['预处理后的灰度图像已保存为: ', output_gray_path]);

    test_wavefront = double(input_image_gray_8bit) / 255; % 作为重建目标 (归一化到 0-1)

catch ME
    warning(['图像预处理失败: ', ME.message]);
    return;
end

%% 3. 模拟衍射
F_object = fft2(test_wavefront);
Fabs_data = abs(F_object);
Fabs_data_shifted = fftshift(Fabs_data);

disp('模拟衍射图样已计算完成。');

%% 4. 定义 GSW 重建参数 (已初步调整)
config = struct(...
    'n1',       500,       ... % HIO迭代次数 (内循环)
    'gen',      20,        ... % GSW世代数 (外循环)
    'n2',       80,        ... % 每世代HIO迭代次数
    'rep',      4,         ... % 并行副本数
    'alpha',    50,        ... % << 已调整：OSS平滑系数，增加以抑制噪声
    'sigma',    1.5,       ... % 高斯模糊标准差 (支持域更新)
    'cutoff1',  0.02,      ... % 支持域更新的下阈值
    'cutoff2',  0.12,      ... % << 已调整：支持域更新的上阈值，增加以更严格地收缩支持域
    'beta',     0.85,      ... % HIO反馈系数
    'checker',  false(256, 256) ... % << 已调整：中央掩膜已移除
);

% 确保 checker 全为 false，不进行中央掩膜操作
% (已通过 config.checker = false(256, 256) 确保，不再需要额外的循环)

fprintf('GSW重建参数配置已设置。\n');
fprintf('  alpha (OSS平滑系数) 已调整为: %d\n', config.alpha);
fprintf('  cutoff2 (支持域收缩阈值) 已调整为: %.2f\n', config.cutoff2);
fprintf('  中央掩膜已禁用。\n');

%% 5. GSW 重建
disp(' ');
disp('------ 开始执行 GSW 算法 ------');
try
    if isempty(gcp('nocreate'))
        parpool('Processes', min(config.rep, feature('numcores')));
    end

    tic;
    % gshrinkwrap 函数现在也会返回 efs_all_reps 和 S_all_gens
    [R_reconstructed_all_reps, S_all_gens, ~, efs_all_reps] = gshrinkwrap(...
        Fabs_data_shifted, ...
        config.n1, ...
        config.checker, ...
        config.gen, ...
        config.n2, ...
        config.rep, ...
        config.alpha, ...
        config.sigma, ...
        config.cutoff1, ...
        config.cutoff2, ...
        config.beta); % 传入 beta 参数
    time_elapsed = toc;

    disp(' ');
    disp(['重建完成! 总耗时: ', num2str(time_elapsed/60, '%.1f'), ' 分钟']);

catch ME
    warning('重建失败！错误信息：%s', ME.message);
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end
    rethrow(ME);
end

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

%% 6. 结果评估 (不进行自动对齐)
best_rep_idx = 1; % 对于任意图像，我们先取第一个副本进行演示，后续可根据收敛曲线选择
R_final_reconstructed = R_reconstructed_all_reps(:,:,best_rep_idx);

% 计算原始重建结果的 EF 和 ER
final_ef = ef(Fabs_data_shifted, fft2(R_final_reconstructed), config.checker);
final_er = er(test_wavefront, abs(R_final_reconstructed), []);
fprintf('\n最终重建结果 (第一个副本):\n  EF误差: %.4f\n  ER误差: %.4f\n', final_ef, final_er);

%% 7. 可视化输出 (包含收敛曲线和最终支撑)
figure('Name', '任意图像重建结果与收敛分析', 'NumberTitle', 'off', 'Position', [50, 50, 1600, 800]);

% 1. 原始灰度图像
subplot(2, 4, 1);
imshow(test_wavefront, []);
title('1. 原始灰度图像');
colormap(gca, gray);

% 2. 衍射图样 (幅度，对数显示)
subplot(2, 4, 2);
imshow(log(Fabs_data_shifted + 1), []);
title('2. 衍射图样 (log(Amplitude + 1))');
colormap(gca, hot);

% 3. 重建结果 (未对齐)
subplot(2, 4, 3);
imshow(abs(R_final_reconstructed), []);
title(['3. 重建结果 (EF=', num2str(final_ef, '%.3f'), ', ER=', num2str(final_er, '%.3f'), ')']);
colormap(gca, jet);
colorbar;

% 4. 重建结果的绝对误差图
subplot(2, 4, 4);
error_map = abs(abs(R_final_reconstructed) - test_wavefront);
imshow(error_map, []);
title('4. 绝对误差图');
colormap(gca, jet);
colorbar;

% 5. 初始支撑 (GSW通常会内部生成，这里显示第一个世代的支撑)
subplot(2, 4, 5);
if ~isempty(S_all_gens) && size(S_all_gens, 3) >= 1
    imshow(S_all_gens(:,:,1), []);
    title('5. 初始支撑 (Generation 1)');
else
    text(0.5, 0.5, '无法获取初始支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 6. 最终收敛支撑
subplot(2, 4, 6);
if ~isempty(S_all_gens) && size(S_all_gens, 3) == config.gen + 1
    imshow(S_all_gens(:,:,config.gen+1), []); 
    title(['6. 最终收敛支撑 (Generation ', num2str(config.gen), ')']);
else
    text(0.5, 0.5, '无法获取最终支撑', 'HorizontalAlignment', 'center');
end
colormap(gca, gray);

% 7. 收敛曲线 (各代最佳EF)
subplot(2, 4, 7);
if ~isempty(efs_all_reps) && size(efs_all_reps, 1) == config.gen + 1
    best_efs_per_gen = min(efs_all_reps, [], 2); % 每代所有副本中的最佳EF
    plot(0:config.gen, best_efs_per_gen, 'b', 'LineWidth', 2); % 从第0代开始
    xlabel('世代数');
    ylabel('最佳EF误差');
    title('7. 重建收敛曲线 (最佳EF)');
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
        title('7. 重建收敛曲线 (部分)');
        grid on;
    end
end

% 8. 留空或未来扩展 (例如，可以显示不同副本的ER曲线)
subplot(2, 4, 8);
text(0.5, 0.5, '此处可用于未来扩展或额外分析', 'HorizontalAlignment', 'center', 'FontSize', 10);
axis off;

% 保存最终结果到工作区
final_reconstruction = abs(R_final_reconstructed);
final_EF_error = final_ef;
final_ER_error = final_er;

disp(' ');
disp('脚本运行完成。重建结果和收敛分析已显示在图窗中，并保存到工作区变量。');