clc; clear; close all;

%% ========== 1. 数据 ==========
x = 1:5;

Y = [
    0.85 0.78 0.72 0.69 0.65;
    0.82 0.75 0.70 0.66 0.62;
    0.80 0.73 0.68 0.64 0.60;
    0.78 0.71 0.66 0.62 0.58;
    0.76 0.69 0.64 0.60 0.56
];

%% ========== 2. 颜色 ==========
C = [
    0.00 0.45 0.74;
    0.85 0.33 0.10;
    0.93 0.69 0.13;
    0.49 0.18 0.56;
    0.47 0.67 0.19
];

%% ========== 3. 分组柱状图 ==========
figure;
hBar = bar(x, Y, 'grouped');
hold on;

for i = 1:length(hBar)
    hBar(i).FaceColor = C(i,:);
    hBar(i).EdgeColor = 'k';
    hBar(i).LineWidth = 1;
end

%% ========== 4. 纹理填充 ==========
hatchfill2(hBar(6),'cross','HatchDensity',40,'HatchColor','k');
hatchfill2(hBar(7),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(hBar(8),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');
hatchfill2(hBar(9),'single','HatchAngle',-45,'HatchDensity',40,'HatchColor','k');
hatchfill2(hBar(10),'cross','HatchDensity',30,'HatchColor','k');

%% ========== 5. 标签 ==========
xlabel('Samples');
ylabel('RMSE (m)');
title('Grouped Bar with Hatch Patterns');

box on;
set(gca,'LineWidth',1);

