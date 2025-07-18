function B = myalign(A, B)
    % 计算 A 和当前 B 之间的互相关
    xcorr1 = fftshift( ifft2( fft2(A) .* conj(fft2(B)) ) );
    % 寻找全局最大值
    [y1, idx1] = max(xcorr1(:));
    [cr1, cc1] = ind2sub(size(xcorr1), idx1); % 将线性索引转换为行/列索引

    % 计算 A 和 180 度旋转后的 B 之间的互相关
    B_rot = rot90(B, 2); % 180度旋转
    xcorr2 = fftshift( ifft2( fft2(A) .* conj(fft2(B_rot)) ) );
    % 寻找全局最大值
    [y2, idx2] = max(xcorr2(:));
    [cr2, cc2] = ind2sub(size(xcorr2), idx2);

    % 比较两个相关峰的高度，选择最佳的对齐方式
    if y2 > y1
        B = B_rot; % 采用旋转后的 B
        r_shift = cr2 - round(size(A,1)/2); % 计算行方向的偏移量
        c_shift = cc2 - round(size(A,2)/2); % 计算列方向的偏移量
        disp(['myalign: 已反转并偏移 (' int2str(r_shift) ', ' int2str(c_shift) ').']);
    else
        r_shift = cr1 - round(size(A,1)/2); % 计算行方向的偏移量
        c_shift = cc1 - round(size(A,2)/2); % 计算列方向的偏移量
        disp(['myalign: 已偏移 (' int2str(r_shift) ', ' int2str(c_shift) ').']);
    end

    % 应用循环移位进行对齐
    B = circshift(B, [r_shift, c_shift]);
end