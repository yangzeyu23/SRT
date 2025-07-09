clearvars;                         % 清除工作区所有变量
close all;clc                      % 关闭所有图窗，清空命令行

addpath('./function');             % 添加自定义函数文件夹到搜索路径

lgraph = layerGraph;               % 创建一个空的layerGraph对象，用于搭建网络结构
inputSize = [2160 3840];           % 输入图片的尺寸（高2160，宽3840）
outputSize = [3840 3840];          % 输出图片的尺寸（高3840，宽3840）

layers = [
    imageInputLayer([inputSize 1],'Normalization','none','Name','input')];
                                    % 定义输入层，输入为2160x3840x1的灰度图，不做归一化，命名为'input'

lgraph = addLayers(lgraph,layers);  % 将输入层添加到layerGraph中

[lgraph,outputName] = unet(lgraph, 'input');
                                    % 调用自定义unet函数，基于输入层构建U-Net结构
                                    % outputName为U-Net最后一层的名字

% odd to even
% lgraph = replaceLayer(lgraph,'ResUpConv3_1',...
%     transposedConv2dLayer(3,64,'Stride',2,'Cropping',[1 1 0 1],'Name','ResUpConv3_1'));
% lgraph = replaceLayer(lgraph,'SkipUpConv3',...
%     transposedConv2dLayer(3,64,'Stride',2,'Cropping',[1 1 0 1],'Name','SkipUpConv3'));
                                    % （注释掉的代码）用于调整U-Net中某些反卷积层的参数，实现奇偶尺寸的对齐

layers = [
    batchNormalizationLayer('Name','BN')    % 批归一化层，命名为'BN'
    tanhpiLayer('tanh')];                   % 自定义tanhpi激活层，命名为'tanh'

lgraph = addLayers(lgraph,layers);          % 添加上述两层到网络
lgraph = connectLayers(lgraph,outputName,'BN');
                                            % 将U-Net输出连接到BN层

lambda = 532e-6;                            % 波长，单位mm（532nm）
z = 160;                                    % 传播距离，单位mm
dp = 0.00374;                               % 像素间距，单位mm
Lx = dp*inputSize(2);                       % 输入图像的物理宽度（mm）
Ly = dp*inputSize(1);                       % 输入图像的物理高度（mm）
[x,y] = meshgrid(-Lx/2:dp:Lx/2-dp,-Ly/2:dp:Ly/2-dp);
                                            % 生成以中心为原点的物理坐标网格
P = pi*(x.^2 + y.^2)/(lambda*z);            % 计算菲涅耳衍射的相位因子P

lgraph = addLayers(lgraph,fresnelLayer(inputSize,outputSize,P,'Name','I'));
                                            % 添加自定义菲涅耳传播层，输入输出尺寸分别为inputSize和outputSize，P为相位因子，命名为'I'
lgraph = connectLayers(lgraph,'tanh','I');  % 将tanh激活层输出连接到菲涅耳层

dlnet = dlnetwork(lgraph);                  % 将layerGraph转换为dlnetwork对象，便于后续训练
save('HoloEncoder_untrained.mat','dlnet');  % 保存未训练的网络到.mat文件
