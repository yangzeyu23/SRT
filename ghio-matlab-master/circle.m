% circle.m

%% 1. 定义图像和光斑参数

image_size = 256; % 
radius_ratio = 0.25; % 圆形光斑半径占图像尺寸一半

%% 2. 生成圆形二值图像

object = zeros(image_size, image_size);

% 计算图像中心坐标
center_x = image_size / 2 + 0.5;
center_y = image_size / 2 + 0.5;

% 计算圆形半径
circle_radius = image_size * radius_ratio; 

% 遍历像素，设置圆形区域为1
for i = 1:image_size
    for j = 1:image_size
        distance = sqrt((i - center_y)^2 + (j - center_x)^2);
        if distance <= circle_radius
            object(i, j) = 1; 
        end
    end
end

disp('圆形二值图像已生成。');

%% 3. 模拟衍射：计算频率域分布强度 (Fabs_data)

F_object = fft2(object);

% 提取频率域分布强度 (幅度) 作为衍射数据
Fabs_data = abs(F_object);

% 将零频率分量（中心）移动到中心
Fabs_data_shifted = fftshift(Fabs_data);

disp('频率域分布强度（衍射图样）已计算。');

%% 4. 定义初始空间域支撑 (Support, S)

%创建一个与物体大小相近的圆形支撑
support_buffer = 10; 
support_radius = circle_radius + support_buffer;
S = zeros(image_size, image_size);

for i = 1:image_size
    for j = 1:image_size
        distance = sqrt((i - center_y)^2 + (j - center_x)^2);
        if distance <= support_radius
            S(i, j) = 1;
        end
    end
end
S = logical(S); % 将支撑转换为逻辑类型

disp('初始空间域支撑已定义。');

%% 5. 执行 HIO 算法进行图像重建

n_iterations = 2000; 
% hio2d 函数的第一个参数通常是Fabs，但在Guided HIO/Shrinkwrap中，
% 可能会先对R进行fft2再传入。这里直接使用Fabs_data。

R_reconstructed = hio2d(Fabs_data, S, n_iterations);

disp(['HIO 重建完成，迭代 ' num2str(n_iterations) ' 次。']);

%% 6. 可视化结果
figure;

subplot(1, 3, 1);
imshow(object, []);
title('原始圆形光斑 (空间域)');
colormap(gca, gray); % 灰度显示

subplot(1, 3, 2);
imshow(log(Fabs_data_shifted + 1), []);    % 对数显示
title('频率域分布强度 (衍射图样, 对数显示)');
colormap(gca, hot); % 热图显示强度

subplot(1, 3, 3);
imshow(abs(R_reconstructed), []); % 显示重建图像的幅度
title('HIO 重建结果 (空间域)');
colormap(gca, gray); % 灰度显示

% 计算频率域误差 (可选，需要 ef.m)
% checker = false(image_size, image_size); % 假设没有需要排除的区域
% final_ef_error = ef(Fabs_data, fft2(R_reconstructed), checker);
% disp(['最终频率域误差 (EF): ' num2str(final_ef_error)]);