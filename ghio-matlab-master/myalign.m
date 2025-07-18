function [aligned_image] = myalign(reference_image, image_to_align)
% MYALIGN: 使用相位相关方法将图像进行对齐。
%   此函数计算 'image_to_align' 相对于 'reference_image' 的像素位移，
%   并应用此位移，使其与参考图像对齐。
%   它利用相位相关方法，该方法对亮度变化不敏感，适用于衍射图像重建中的对齐任务。

    % 获取图像尺寸，用于后续的零填充和结果裁剪。
    [rows, cols] = size(reference_image);

    % 进行零填充以避免傅里叶变换中的周期性卷绕伪影（wraparound artifacts）。
    % 填充到至少两倍原图尺寸，并且填充后的尺寸最好是2的幂次方，
    % 以优化FFT算法的计算效率。
    padded_rows = 2^nextpow2(rows * 2);
    padded_cols = 2^nextpow2(cols * 2);

    % 计算参考图像和待对齐图像的傅里叶变换（应用零填充）。
    FA = fft2(reference_image, padded_rows, padded_cols);
    FB = fft2(image_to_align, padded_rows, padded_cols);

    % 计算互功率谱 (Cross-Power Spectrum)。
    % 互功率谱的逆傅里叶变换其峰值表示了图像的位移。
    % 公式: R_AB = F_A^* * F_B / |F_A^* * F_B|
    % 添加一个非常小的正数 (eps) 到分母中，以防止在分母为零时发生除法错误。
    cross_power_spectrum = (conj(FA) .* FB) ./ (abs(conj(FA) .* FB) + eps);

    % 对互功率谱进行逆傅里叶变换，得到相关图。
    % 相关图中的峰值位置指示了两幅图像之间的精确像素位移。
    % real() 函数用于处理由于浮点计算精度问题可能产生的微小虚部，确保结果为实数图像。
    correlation_map = real(ifft2(cross_power_spectrum));

    % 寻找相关图中的峰值位置。
    % 峰值最大值对应的坐标即为位移量。
    [~, max_idx] = max(correlation_map(:)); % 找到最大值及其在数组中的线性索引
    [y_peak, x_peak] = ind2sub(size(correlation_map), max_idx); % 将线性索引转换为行列坐标

    % 计算图像的像素位移量 (shift_y, shift_x)。
    % 逆傅里叶变换的结果默认以 (1,1) 作为零位移点。
    % 如果峰值位于填充图像的下半部分或右半部分，则表示负向位移。
    shift_y = y_peak - 1; % 初始计算从1开始的位移量
    if shift_y > padded_rows / 2
        shift_y = shift_y - padded_rows; % 转换为负位移，处理周期性边界效应
    end

    shift_x = x_peak - 1; % 初始计算从1开始的位移量
    if shift_x > padded_cols / 2
        shift_x = shift_x - padded_cols; % 转换为负位移
    end

    % 将计算出的位移量应用到原始大小的待对齐图像上。
    % MATLAB 的 circshift 函数可以高效地实现图像的周期性像素位移。
    % 注意：这里的 circshift 是对原始图像尺寸操作，而非填充后的图像。
    aligned_image = circshift(image_to_align, [shift_y, shift_x]);

end