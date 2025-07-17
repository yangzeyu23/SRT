%grayscale_test.m

%% 1. 加载并预处理输入图像作为测试波前


image_filename = '0805.png'; 

% 如果你没有现成的图片，Matlab可以生成一个测试图片，例如：
% img_raw = imread('cameraman.tif'); % Matlab自带的灰度图
% 或者自己创建一个简单的梯度图：
% img_raw = repmat(linspace(0, 1, 512), 512, 1); 
% img_raw = im2uint8(img_raw); % 转换为uint8类型

try
    img_raw = imread(image_filename);
    disp(['成功加载图像: ' image_filename]);
catch
    warning(['无法加载图像 "' image_filename '"。将使用一个默认的灰度测试图像。']);
    img_raw = imread('cameraman.tif'); % 使用Matlab自带的cameraman图像作为默认
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
% 这样，我们的测试波前在物理上就是由16个离散幅度值构成的
num_gray_levels = 16; % 2^4 = 16 级
object_quantized_normalized = round(object_normalized * (num_gray_levels - 1)) / (num_gray_levels - 1);

% 这是我们真正的“测试波前”（空间域物体）
test_wavefront = object_quantized_normalized;

disp(['测试波前已准备完成，尺寸 ' num2str(image_size) 'x' num2str(image_size) '，包含 ' num2str(num_gray_levels) ' 个灰度级。']);

%% 2. 模拟衍射：计算频率域分布强度 (Fabs_data)

% 仍然使用夫琅禾费衍射近似，即直接进行傅里叶变换
F_test_wavefront = fft2(test_wavefront);

% 提取频率域分布强度 (幅度) 作为衍射数据
Fabs_data = abs(F_test_wavefront);

% 将零频率分量（中心）移动到图像中心，以便可视化
Fabs_data_shifted = fftshift(Fabs_data);

disp('频率域分布强度（衍射图样）已计算。');

%% 3. 定义初始空间域支撑 (Support, S)
% 对于一般图像，一个简单的全尺寸支撑通常是一个好的开始，
% 或者可以基于非零像素来定义一个更紧密的支撑。
% 这里我们先使用一个包含所有可能物体区域的简单全尺寸支撑。
S = true(image_size, image_size); 

% 如果物体可能只占图像一部分，可以考虑更智能的支撑：
% S = (test_wavefront > (max(test_wavefront(:))/10)); % 简单阈值，去除极小值

disp('初始空间域支撑已定义。');

%% 4. 执行 HIO 算法进行图像重建
n_iterations = 10000; % 增加迭代次数以更好地收敛多灰度级图像

% 调用 hio2d 函数进行重建
R_reconstructed = hio2d(Fabs_data, S, n_iterations);

disp(['HIO 重建完成，迭代 ' num2str(n_iterations) ' 次。']);

%% 5. 可视化结果
figure;

subplot(1, 3, 1);
imshow(test_wavefront, []);
title('原始测试波前 (空间域)');
colormap(gca, gray);

subplot(1, 3, 2);
imshow(log(Fabs_data_shifted + 1), []); % 对数显示强度，以便看清细节
title('频率域分布强度 (衍射图样, 对数显示)');
colormap(gca, hot);

subplot(1, 3, 3);
imshow(abs(R_reconstructed), []); % 显示重建图像的幅度
title('HIO 重建结果 (空间域)');
colormap(gca, gray);

%%计算并显示频率域误差 (需要 ef.m)
checker = false(image_size, image_size); % 假设没有需要排除的区域
final_ef_error = ef(Fabs_data, fft2(R_reconstructed), checker);
disp(['最终频率域误差 (EF): ' num2str(final_ef_error)]);

%% 计算空间域误差 (需要 er.m)
space_error = er(test_wavefront, R_reconstructed, S);
disp(['空间域误差 (ER): ' num2str(space_error)]);
disp('测试完成。');