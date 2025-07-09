clearvars -except dlnet;           % 清除除dlnet外的所有变量
close all;                         % 关闭所有图窗
clc;                               % 清空命令行

addpath('./function');             % 添加自定义函数文件夹到搜索路径
load HoloEncoder_trained.mat       % 加载训练好的网络

X = imread('./images/0805.png');    % 读取图片
X = im2gray(X);                    % 转为灰度图
X = imresize(X,[2160,3840]);       % 调整图片尺寸为2160x3840
X = single(X);                     % 转为单精度浮点型
[m,n] = size(X);                   % 获取图片尺寸

dlX = gpuArray(dlarray(X,'SSCB')); % 转为深度学习数组并放到GPU，格式为'SSCB'

tic                                % 计时开始
dlY = forward(dlnet,dlX,'Outputs','tanh'); % 前向推理，输出tanh层结果（全息图）
toc;                               % 计时结束

dlZ = forward(dlnet,dlX);          % 前向推理，输出最终重建结果

Y = gather(extractdata(dlY));      % 提取数据并从GPU转回CPU
Z = gather(extractdata(dlZ));      % 提取数据并从GPU转回CPU

figure,imshow(Y,[]);title('hologram')      % 显示全息图
figure,imshow(Z,[]);title('reconstruction')% 显示重建图像

