clc;        
clear all;  
close all;  

%% --- 定义参数 ---

lambda = 532e-9; 
z = 10e-3;       
N = 256;         
spot_diameter_physical = 0.5e-3; 
spot_to_image_ratio = 0.5;

field_size_physical = spot_diameter_physical / spot_to_image_ratio;
% 仿真区域的物理边长

dx = field_size_physical / N; 
% 单个像素对应的物理尺寸 (dx = dy)

fprintf('--------- 仿真中各物理参数 ---------\n');
fprintf('入射光斑物理直径:    %.3f mm\n', spot_diameter_physical * 1e3);
fprintf('仿真区域物理边长 (L): %.3f mm\n', field_size_physical * 1e3);
fprintf('单个像素对应的物理尺寸 : %.3f µm \n', dx * 1e6);
fprintf('总像素数 (N x N):   %d x %d\n', N, N);

fprintf('----------------------------------\n\n');


%% --- 入射光场 (圆形光斑) ---

x = ((-N/2 : N/2-1) * dx); 
y = ((-N/2 : N/2-1) * dx); 
[X, Y] = meshgrid(x, y);

aperture_radius = spot_diameter_physical / 2; 
input_field_amplitude = double( (X.^2 + Y.^2) <= aperture_radius^2 );

input_field_phase = zeros(N, N);  % 设初始相位为0
input_field = input_field_amplitude .* exp(1j * input_field_phase); % 入射复振幅

% 可视化
figure('Name', '入射光斑');
imagesc(x*1e3, y*1e3, input_field_amplitude); 
axis square;          
colormap(gray);       
title('入射光斑 (复振幅幅度)');
xlabel('X 坐标 (mm)');
ylabel('Y 坐标 (mm)');
colorbar;
set(gca, 'FontSize', 12);
saveas(gcf, 'input_field_amplitude.png');


%% --- 角谱衍射 ---

df = 1 / field_size_physical; 
fx = ((-N/2 : N/2-1) * df);  
fy = ((-N/2 : N/2-1) * df);  
[Fx, Fy] = meshgrid(fx, fy);

% 传递函数
k_term_squared = (1/lambda)^2 - Fx.^2 - Fy.^2;
kz = sqrt(k_term_squared);
H = exp(1j * 2 * pi * z * kz);


U0_freq = fftshift(fft2(input_field)); 

Uz_freq = U0_freq .* H; 

Uz = ifft2(ifftshift(Uz_freq)); 


%% --- 计算光强分布 ---

Intensity_z = abs(Uz).^2;

% 归一化
max_intensity = max(Intensity_z(:));

% 计算衍射最大光强比原始光强增大了多少倍, 原始光强在光斑区域设为 1
original_peak_intensity = 1;
intensity_increase_factor = max_intensity / original_peak_intensity;

Intensity_z_normalized = Intensity_z / max_intensity;

fprintf('衍射光强分布原始最大值 (未归一化): %e\n', max_intensity);
fprintf('原始入射光强 (光斑内): %d\n', original_peak_intensity);
fprintf('当前条件下，衍射图样最大光强比原始光强增大了约 %.2f 倍。\n', intensity_increase_factor);
fprintf('衍射光强最大值归一化为: %.1f\n', max(Intensity_z_normalized(:)));


%% --- 可视化 ---

figure('Name', ['衍射光强分布 (距离 ', num2str(z*1e3), ' mm)']);

imagesc(x*1e3, y*1e3, Intensity_z_normalized); 
axis square;
colormap(gray); 
colorbar;

title('归一化光强分布');
xlabel('X 坐标 (mm)');
ylabel('Y 坐标 (mm)');
set(gca, 'FontSize', 10);

saveas(gcf, 'diffraction_intensity_distribution.png');