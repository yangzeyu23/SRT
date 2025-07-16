% grayscale_shrinkwrap.m

%% 1. 加载并预处理输入图像作为测试波前

image_filename = '0805.png'; 

%备用：img_raw = imread('cameraman.tif');

% 尝试加载图像
try
    img_raw = imread(image_filename);
    disp(['成功加载图像: ' image_filename]);
catch
    warning(['无法加载图像 "' image_filename '"。将使用一个默认的灰度测试图像（cameraman.tif）。']);
    img_raw = imread('cameraman.tif'); 
end

% 转换为灰度图像（如果原始图像是彩色的）
if size(img_raw, 3) == 3
    img_gray = rgb2gray(img_raw);
else
    img_gray = img_raw;
end

% 将图像缩放为 256x256 像素
image_size = 256;
object_original = imresize(img_gray, [image_size, image_size]);

% 将图像转换为 4-bit 灰度（16个灰度级：0到15）
% 确保数据类型是double，并归一化到0-1范围，便于傅里叶变换
% 原始图像通常是uint8 (0-255)，所以先归一化到0-1
object_normalized = double(object_original) / 255;

% 量化到16个灰度级（0到15），然后再次归一化到0-1，作为测试波前的幅度
num_gray_levels = 16; % 2^4 = 16 级
test_wavefront = round(object_normalized * (num_gray_levels - 1)) / (num_gray_levels - 1);

disp(['测试波前已准备完成，尺寸 ' num2str(image_size) 'x' num2str(image_size) '，包含 ' num2str(num_gray_levels) ' 个灰度级。']);

%% 2. 模拟衍射：计算频率域分布强度 (Fabs_data)

% 仍然使用夫琅禾费衍射近似（远场）
F_test_wavefront = fft2(test_wavefront);
Fabs_data = abs(F_test_wavefront);
Fabs_data_shifted = fftshift(Fabs_data);

disp('频率域分布强度（衍射图样）已计算。');

%% 3. 定义初始空间域支撑 (Support, S)

% 对于shrink-wrap，初始支撑可以从自相关图生成，或者给一个宽泛的估计。
% 这里我们先给一个略大于物体实际内容的矩形支撑作为初始S，
% 或者可以根据非零像素自动生成一个紧密的初始支撑。
% 为了简单起见，我们先用一个包含大部分物体的矩形支撑。
% 你也可以尝试使用一个基于物体非零像素的更紧密的支持：
initial_S = (test_wavefront > (max(test_wavefront(:))/5)); % 简单阈值，去除背景的微小噪声
initial_S = imfill(initial_S, 'holes'); % 填充内部空洞

% 如果你的图像物体可能占不满中心区域，可以考虑用 hiosupport 创建一个。
% 例如：support_size_pixels = [180, 180]; % 假设物体大概在180x180区域内
% S_initial_hiosupport = hiosupport([image_size, image_size], support_size_pixels);
% S = S_initial_hiosupport;

S1 = initial_S; % 将初始支撑赋值给 S1

disp('初始空间域支撑已定义。');

%% 4. 执行 SHRINK-WRAP 算法进行图像重建

% shrink-wrap 参数设置
n_hio_initial = 500; % 首次HIO迭代次数
num_generations = 50; % Shrink-Wrap 世代数（支撑更新次数）
n_hio_per_gen = 200;  % 每代Shrink-Wrap内部的HIO迭代次数
alpha_oss = [];       % Oversampling Smoothness (OSS) 参数，[]表示禁用OSS
checker_mask = false(image_size, image_size); % 默认没有需要忽略的频率域区域

% 调用 shrinkwrap 函数进行重建
% [R, Sup, M] = shrinkwrap(Fabs, n, checker, gen, n2, varargin)
% varargin: {alpha, sigma, cutoff1, cutoff2}
% alpha: OSS参数
% sigma: 高斯平滑核的标准差，用于生成新的支撑 (默认3)
% cutoff1: 用于从自相关图生成初始支撑的阈值 (默认0.04)
% cutoff2: 用于从平滑图像生成新支撑的阈值 (默认0.2)

% 可以调整sigma和cutoff2来优化支撑的生成
sigma_sw = 3; % 缩小该值可以使支撑更精细，增大则更平滑
cutoff2_sw = 0.2; % 增大该值可以使支撑更紧凑，减小则更宽松

disp(['开始执行 SHRINK-WRAP 重建，共 ' num2str(num_generations) ' 代...']);

[R_reconstructed_all_gens, S_all_gens, M_all_gens] = shrinkwrap(...
    Fabs_data, n_hio_initial, checker_mask, num_generations, n_hio_per_gen, ...
    alpha_oss, sigma_sw, [], cutoff2_sw);

% 最终重建结果是最后一世代的R
R_final_reconstructed = R_reconstructed_all_gens(:,:,end);

disp('SHRINK-WRAP 重建完成。');

%% 5. 可视化结果

figure;

subplot(1, 3, 1);
imshow(test_wavefront, []);
title('原始测试波前 (空间域)');
colormap(gca, gray);

subplot(1, 3, 2);
imshow(log(Fabs_data_shifted + 1), []); % 对数显示强度
title('频率域分布强度 (衍射图样, 对数显示)');
colormap(gca, hot);

subplot(1, 3, 3);
imshow(abs(R_final_reconstructed), []); % 显示最终重建图像的幅度
title('SHRINK-WRAP 重建结果 (空间域)');
colormap(gca, gray);

% 可选：可视化最后一代的支撑
figure;
imshow(S_all_gens(:,:,end), []);
title('最终收敛的支撑');
colormap(gca, gray);

% 可选：计算并显示频率域误差 (需要 ef.m)
% 注意：HIO算法的Fabs参数是幅度，不是强度，所以如果输入的是强度，需要开平方根
Fabs_for_ef = sqrt(Fabs_data); % ef函数期望幅度
final_ef_error = ef(Fabs_for_ef, fft2(R_final_reconstructed), checker_mask);
disp(['最终频率域误差 (EF): ' num2str(final_ef_error)]);