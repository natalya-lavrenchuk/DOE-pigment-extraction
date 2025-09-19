

% Name: Natalya Lavrenchuk (ID: 2141882)
% Course: SP25 Machine Learning for Process Engineering
% Instructor: Professor Pierantonio Facco
% Assignment: Homework #2 – DoE
% Date: June 19, 2025

% The purpose of this homework is to apply DoE principles and linear 
% regression techniques to build an empirical model on industrial pigment 
% extraction data for process optimization purposes. 




%% ---- Question 1 - data visualization

figure('Color','w');
hold on;
grid on;
view(3);

% Plot the edges of the unit cube:
edges = [
    -1  -1  -1;   +1  -1  -1;   +1  +1  -1;   -1  +1  -1;   -1  -1  -1;  % bottom face
    -1  -1  +1;   +1  -1  +1;   +1  +1  +1;   -1  +1  +1;   -1  -1  +1;  % top face
    +1  -1  -1;   +1  -1  +1;   +1  +1  +1;   +1  +1  -1;   % front vertical edge
    -1  -1  -1;   -1  -1  +1;   -1  +1  +1;   -1  +1  -1   % back vertical edge
];
plot3(edges(:,1), edges(:,2), edges(:,3), ...
      'Color',[0.7 0.7 0.7], 'LineWidth',1.5);

% Plot each run in the 3D space:

edge_col   = "blue";   marker_edge   = 'o';
center_col = "red";   marker_center = 'o';

N = size(Xc,1);
for i = 1:N
    x = Xc(i,1); 
    y = Xc(i,2);
    z = Xc(i,3);
    
    if all([x,y,z] == 0)
        % Center‐point replicate
        scatter3(x, y, z, 120, center_col, marker_center, 'filled');
    else
        % Edge or face‐midpoint run
        scatter3(x, y, z,  80, edge_col,   marker_edge,   'filled');
    end
end

% 3) Cosmetics: labels, ticks, legend, axes limits, etc.
xlabel('Temperature');
ylabel('Time');
zlabel('Mass');

xticks([-1  0  1]);
yticks([-1  0  1]);
zticks([-1  0  1]);

xlim([-1.2  1.2]);
ylim([-1.2  1.2]);
zlim([-1.2  1.2]);

axis equal;   % Keep the cube proportions correct

title('PigmentOne Experiment Design');
hold off;




%% ---- Question 2 - Main effects plots

% main effects plot for Betacyanin
figure;
maineffectsplot( Y(:,1), X, 'varnames',{'Temp. [deg C]','Time [min]','Mass [g]'} );

%  main effects plot for Betaxanthin
figure;
maineffectsplot( Y(:,2), X, 'varnames',{'Temp. [deg C]','Time [min]','Mass [g]'} );





%% ---- Question 3 - Interaction plots

% interaction plots for betacyanin
figure;
interactionplot( Y(:,1), X, 'varnames',{'Temp. [deg C]','Time [min]','Mass [g]'} );

% interaction plots for betaxanthin
figure;
interactionplot( Y(:,2), X, 'varnames',{'Temp. [deg C]','Time [min]','Mass [g]'} );





%% ---- Question 4 - Build the regression model and visualize it

%% ---- Question 4a-b - estimate the regression models

% ANOVA for betacyanin
tbl_cyan = array2table([Xc, Y(:,1)], ...
      'VariableNames',{'Temp','Time','Mass','Betacyanin'});
mdl_cyan = fitlm(tbl_cyan, ...
    'Betacyanin ~ Temp + Time + Mass + Temp:Time + Temp:Mass + Time:Mass + Temp^2 + Time^2 + Mass^2' );
anovaTbl_cyan = anova(mdl_cyan);
disp(anovaTbl_cyan)

TempCat = categorical( Xc(:,1), [-1 0 1], {'30 °C','40 °C','50 °C'} );
TimeCat = categorical( Xc(:,2), [-1 0 1], {'20 min','70 min','120 min'} );
MassCat = categorical( Xc(:,3), [-1 0 1], {'0.5 g','1 g','1.5 g'} );

% Plot the main effects
[p,tbl,stats] = anovan(Y(:,1),{TempCat, TimeCat, MassCat},'model','linear','display','off','varnames',{'Temp','Time','Mass'});
figure;multcompare(stats,'dimension',1);
figure;multcompare(stats,'dimension',2);
figure;multcompare(stats,'dimension',3);

% ANOVA for betaxanthin
tbl_xan = array2table([Xc, Y(:,2)], ...
      'VariableNames',{'Temp','Time','Mass','Betaxanthin'});
mdl_xan = fitlm(tbl_xan, ...
    'Betaxanthin ~ Temp + Time + Mass + Temp:Time + Temp:Mass + Time:Mass + Temp^2 + Time^2 + Mass^2' );
anovaTbl_xan = anova(mdl_xan);
disp(anovaTbl_xan)

% Plot the main effects
[p,tbl,stats] = anovan(Y(:,2),{TempCat, TimeCat, MassCat},'model','linear','display','off','varnames',{'Temp','Time','Mass'});
figure;multcompare(stats,'dimension',1);
figure;multcompare(stats,'dimension',2);
figure;multcompare(stats,'dimension',3);


%prepare for regression
x1 = Xc(:,1);
x2 = Xc(:,2);
x3 = Xc(:,3);
N  = size(Xc,1);

XX = [ ...
   ones(N,1), ...               % intercept
   x1, x2, x3, ...              % main effects
   x1.*x2, x1.*x3, x2.*x3, ...  % two‐factor interactions
   x1.^2,  x2.^2,  x3.^2        % quadratic terms
];

% regression for betacyanin
[b_cyan,bint_cyan,r_cyan,rin_cyan,stats_cyan] = regress( Y(:,1), XX );
% regression for betaxanthin
[b_xan,bint_xan,r_xan,rin_xan,stats_xan] = regress( Y(:,2), XX );

% combine the regression coefficients b1 and b2 into variable b and
% uncertainty coefficients into b_unc
b = [b_cyan, b_xan];
b_unc = [bint_cyan, bint_xan];

% plot the regression coefficients for betacyanin
figure;bar(b_cyan(2:end,1));hold on;errorbar(b_cyan(2:end,1),(bint_cyan(2:end,2)-bint_cyan(2:end,1))./2,'.');
labels = {'x1','x2','x3','x1*x2','x1*x3','x2*x3','x1^2','x2^2','x3^2'};
xticks(1:9);
xticklabels(labels);
title("Regression Coefficients for Betacyanin");xlabel('effect');ylabel('regression coefficients');

% plot the regression coefficients for betaxanthin
figure;bar(b_xan(2:end,1));hold on;errorbar(b_xan(2:end,1),(bint_xan(2:end,2)-bint_xan(2:end,1))./2,'.');
labels = {'x1','x2','x3','x1*x2','x1*x3','x2*x3','x1^2','x2^2','x3^2'};
xticks(1:9);
xticklabels(labels);
title("Regression Coefficients for Betaxanthin");xlabel('effect');ylabel('regression coefficients');





%% ---- Question 4c - check model adequacy

% save the individual R^2 values and residuals in variable R2 and r
R2 = [stats_cyan(1), stats_xan(1)];
r = [r_cyan, r_xan];
yhat_cyan = XX * b_cyan;
yhat_xan = XX * b_xan;

% Residual plots for Betacyanin
figure('Position',[100 100 800 600]);
subplot(2,3,1), qqplot(r_cyan), title('Normal Q–Q Plot');
subplot(2,3,2), scatter(yhat_cyan,r_cyan,'filled'), xlabel('Residuals'), ylabel('Fitted y'), title('Residual vs Fitted Values');
subplot(2,3,3), plot(1:length(r_cyan),r_cyan, '-o'), xlabel('Time Order'); ylabel('Residuals'), title('Residuals over Runs');
subplot(2,3,4), scatter(X(:,1),r_cyan,[],'filled'), xlim([25 55]), ylim([-0.15, 0.15]), xlabel('Temperature'), ylabel('Residuals'), title('Residuals vs Temperature');
subplot(2,3,5), scatter(X(:,2),r_cyan,[],'filled'), xlim([0 150]), ylim([-0.15, 0.15]), xlabel('Time'),  ylabel('Residuals'), title('Residuals vs Time');
subplot(2,3,6), scatter(X(:,3),r_cyan,[],'filled'), xlim([0.25 1.75]), ylim([-0.15, 0.15]), xlabel('Mass'); ylabel('Residuals'), title('Residuals vs Mass');


% Residual plots for Betaxanthin
figure('Position',[100 100 800 600]);
subplot(2,3,1), qqplot(r_xan), title('Normal Q–Q Plot');
subplot(2,3,2), scatter(yhat_xan,r_xan,'filled'), xlabel('Residuals'), ylabel('Fitted y'), title('Residual vs Fitted Values');
subplot(2,3,3), plot(1:length(r_xan),r_xan, '-o'), xlabel('Time Order'); ylabel('Residuals'), title('Residuals over Runs');
subplot(2,3,4), scatter(X(:,1),r_xan,[],'filled'), xlim([25 55]), ylim([-0.3, 0.3]), xlabel('Temperature'), ylabel('Residuals'), title('Residuals vs Temperature');
subplot(2,3,5), scatter(X(:,2),r_xan,[],'filled'), xlim([0 150]), ylim([-0.3, 0.3]), xlabel('Time'),  ylabel('Residuals'), title('Residuals vs Time');
subplot(2,3,6), scatter(X(:,3),r_xan,[],'filled'), xlim([0.25 1.75]), ylim([-0.3, 0.3]), xlabel('Mass'); ylabel('Residuals'), title('Residuals vs Mass');




%% ---- Question 4d - refine and update the model structure

% check which terms are insignificant to the betacyanin and betaxanthin model
insig_cyan = find( bint_cyan(:,1) < 0 & bint_cyan(:,2) > 0 );

% remove the insignificant coeff from the betacyanin model and plot them
keepCols_cyan    = setdiff(1:size(XX,2), insig_cyan);   
XX_reduced_cyan  = XX(:, keepCols_cyan);
[b_cyan2,bint_cyan2,r_cyan2,rint_cyan2,stats_cyan2] = regress(Y(:,1), XX_reduced_cyan);

figure;bar(b_cyan2(2:end,1));hold on;errorbar(b_cyan2(2:end,1),(bint_cyan2(2:end,2)-bint_cyan2(2:end,1))./2,'.');
labels = {'x2','x3','x2*x3','x1^2','x2^2','x3^2'};
xticks(1:6);
xticklabels(labels);
title("Reduced Reg Coefficients for Betacyanin");xlabel('effect');ylabel('regression coefficients');

% check that the new coefficients are all significant
insig_cyan2 = find(bint_cyan2(:,1) < 0 & bint_cyan2(:,2) > 0);


% remove the insignificant coefficients from the betaxathin model
insig_xan = find( bint_xan(:,1) < 0 & bint_xan(:,2) > 0 );
keepCols_xan    = setdiff(1:size(XX,2), insig_xan);   
XX_reduced_xan  = XX(:, keepCols_xan);
[b_xan2,bint_xan2,r_xan2,rint_xan2,stats_xan2] = regress(Y(:,2), XX_reduced_xan);

insig_xan2 = find( bint_xan2(:,1) < 0 & bint_xan2(:,2) > 0 );

figure;bar(b_xan2(2:end,1));hold on;errorbar(b_xan2(2:end,1),(bint_xan2(2:end,2)-bint_xan2(2:end,1))./2,'.');
labels = {'x1', 'x2','x3', 'x1*x3', 'x2*x3','x1^2','x2^2','x3^2'};
xticks(1:8);
xticklabels(labels);
title("Reduced Reg Coefficients for Betaxanthin");xlabel('effect');ylabel('regression coefficients');


XX_cyan = [ ...
   ones(N,1), ...               % intercept
   x2, x3, ...                  % main effects
   x2.*x3, ...                  % two‐factor interactions
   x1.^2,  x2.^2,  x3.^2        % quadratic terms
];

XX_xan = [ ...
   ones(N,1), ...               % intercept
   x1, x2, x3, ...              % main effects
   x1.*x3, x2.*x3, ...          % two‐factor interactions
   x1.^2,  x2.^2,  x3.^2        % quadratic terms
];

yhat_cyan2 = XX_cyan * b_cyan2;
yhat_xan2 = XX_xan * b_xan2;

% Residual plots for Betacyanin
figure('Position',[100 100 800 600]);
subplot(2,3,1), qqplot(r_cyan2), title('Normal Q–Q Plot');
subplot(2,3,2), scatter(yhat_cyan,r_cyan2,'filled'), xlabel('Residuals'), ylabel('Fitted y'), title('Residual vs Fitted Values');
subplot(2,3,3), plot(1:length(r_cyan2),r_cyan2, '-o'), xlabel('Time Order'); ylabel('Residuals'), title('Residuals over Runs');
subplot(2,3,4), scatter(X(:,1),r_cyan2,[],'filled'), xlim([25 55]), ylim([-0.15, 0.15]), xlabel('Temperature'), ylabel('Residuals'), title('Residuals vs Temperature');
subplot(2,3,5), scatter(X(:,2),r_cyan2,[],'filled'), xlim([0 150]), ylim([-0.15, 0.15]), xlabel('Time'),  ylabel('Residuals'), title('Residuals vs Time');
subplot(2,3,6), scatter(X(:,3),r_cyan2,[],'filled'), xlim([0.25 1.75]), ylim([-0.15, 0.15]), xlabel('Mass'); ylabel('Residuals'), title('Residuals vs Mass');


% Residual plots for Betaxanthin
figure('Position',[100 100 800 600]);
subplot(2,3,1), qqplot(r_xan2), title('Normal Q–Q Plot');
subplot(2,3,2), scatter(yhat_xan2,r_xan2,'filled'), xlabel('Residuals'), ylabel('Fitted y'), title('Residual vs Fitted Values');
subplot(2,3,3), plot(1:length(r_xan2),r_xan2, '-o'), xlabel('Time Order'); ylabel('Residuals'), title('Residuals over Runs');
subplot(2,3,4), scatter(X(:,1),r_xan2,[],'filled'), xlim([25 55]), ylim([-0.3, 0.3]), xlabel('Temperature'), ylabel('Residuals'), title('Residuals vs Temperature');
subplot(2,3,5), scatter(X(:,2),r_xan2,[],'filled'), xlim([0 150]), ylim([-0.3, 0.3]), xlabel('Time'),  ylabel('Residuals'), title('Residuals vs Time');
subplot(2,3,6), scatter(X(:,3),r_xan2,[],'filled'), xlim([0.25 1.75]), ylim([-0.3, 0.3]), xlabel('Mass'); ylabel('Residuals'), title('Residuals vs Mass');

% store the reduced model coefficients, uncertainty, R2, and residuals
b2      = { b_cyan2,      b_xan2      };
b_unc2  = { bint_cyan2,   bint_xan2   };
R2_2    = { stats_cyan2(1) ,stats_xan2(1)  };
r2      = { r_cyan2,      r_xan2      };



%% ---- Question 4e - Build response surface and contour plots

%% Build the response surface and countour plot for Betacyanin
f1 = @(b, x1, x2, x3) ...
     b(1) ...                % intercept
   +       0.*x1 ...         % no x1 term
   +   b(2).*x2 ...          % x2 (Time)
   +   b(3).*x3 ...          % x3 (Mass)
   +   0.*(x1.*x2) ...       % no x1*x2
   +   0.*(x1.*x3) ...       % no x1*x3
   + b(4).*x2.*x3 ...        % x2*x3
   + b(5).*x1.^2 ...         % x1^2 (Temp²)
   + b(6).*x2.^2 ...         % x2^2 (Time²)
   + b(7).*x3.^2;            % x3^2 (Mass²)

figure('Color','w','Units','normalized','Position',[.1 .1 .8 .8]);

% build the x1–x2 grid

npts = 101;
x1v = linspace(-1,1,npts);   % coded Temp
x2v = linspace(-1,1,npts);   % coded Time
[X1g,X2g] = meshgrid(x1v,x2v);
% evaluate on that grid at x3 = 0
Z = f1(b_cyan2, X1g, X2g, 0);
% 3D surface + actual data
subplot(3,2,1);
h = surf(X1g, X2g, Z, 'EdgeColor','none', 'FaceAlpha',0.7);
hold on;
scatter3( ...
  Xc(:,1), Xc(:,2), Y(:,1), ...  % (Temp, Time, actual betacyanin)
  80, 'r', 'filled', 'MarkerEdgeColor','k' ...
);
view(45,30);
xlabel('Temp');   ylabel('Time');   zlabel('Betacyanin (mg/100g)');
title('Response Surface: Temp vs Time @ Mass=1g');
colorbar;
hold off;
% 2D contour + points
subplot(3,2,2);
contour(X1g, X2g, Z, 10, 'ShowText','on','LineWidth',1.2,'LabelFormat','%.1f');
hold on;
scatter( ...
  Xc(:,1), Xc(:,2), ...     % just the (Temp,Time)
  80, 'r', 'filled', 'MarkerEdgeColor','k' ...
);
xlabel('Temp'); ylabel('Time');
title('Contour: Temp vs Time @ Mass=1g');
colorbar;

% make a grid for x1 and x3 from -1 to +1
npts = 101;
x1v = linspace(-1, +1, npts);
x3v = linspace(-1, +1, npts);
[x1g, x3g] = meshgrid(x1v, x3v);

% evaluate the model on the grid
Z = f1(b_cyan2, x1g, 0, x3g);

% 3D surface plot
subplot(3,2,3);
surf(x1g, x3g, Z, 'EdgeColor','none');hold on;
scatter3(Xc(:,1),Xc(:,3),Y(:,1),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Mass');
zlabel('Betacyanin (mg/100g)');
title('Response Surface: Temp vs Mass at Time=70min');
colorbar;
view(45,30);

% 2D contour plot
subplot(3,2,4);
contour(x1g, x3g, Z, 10, ...           
        'ShowText','on', ...           
        'LineWidth',1.2, ...           
        'LabelFormat','%.1f' );hold on;
scatter(Xc(:,1),Xc(:,3),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Mass');
title('Response Surface: Temp vs Mass at Time=70min');
colorbar;

% make a grid for x2 and x3 from -1 to +1
npts = 101;
x2v = linspace(-1, +1, npts);
x3v = linspace(-1, +1, npts);
[x2g, x3g] = meshgrid(x2v, x3v);

% evaluate the model on the grid
Z = f1(b_cyan2, 0, x2g, x3g);

% 3D surface plot
subplot(3,2,5);
surf(x2g, x3g, Z, 'EdgeColor','none');hold on;
scatter3(Xc(:,2),Xc(:,3),Y(:,1),50,'r','filled','MarkerEdgeColor','r');
xlabel('Time');
ylabel('Mass');
zlabel('Betacyanin (mg/100g)');
title('Response Surface: Time vs Mass at Temp = 40degC');
colorbar;
view(45,30);

% 2D contour plot
subplot(3,2,6);
contour(x2g, x3g, Z, 10, ...           
        'ShowText','on', ...           
        'LineWidth',1.2, ...           
        'LabelFormat','%.1f' );hold on;
scatter(Xc(:,2),Xc(:,3),50,'r','filled','MarkerEdgeColor','r');
xlabel('Time');
ylabel('Mass');
title('Response Surface: Time vs Mass at Temp= 40degC');
colorbar;







%% Surface and Contour plots for Betaxanthin model
f2 = @(b_xan2,x1,x2,x3) ...
     b_xan2(1) ...              % intercept
   + b_xan2(2).*x1  ...         % x1
   + b_xan2(3).*x2  ...         % x2
   + b_xan2(4).*x3  ...         % x3
   + 0.*x1.*2 ...               % no x1*x2
   + b_xan2(5).*x1.*x3  ...     % x1*x3
   + b_xan2(6).*x2.*x3 ...      % x2*x3
   + b_xan2(7).*x1.^2 ...       % x1^2
   + b_xan2(8).*x2.^2 ...       % x2^2
   + b_xan2(9).*x3.^2;          % x3^2

figure('Color','w','Units','normalized','Position',[.1 .1 .8 .8]);
npts = 101;
v = linspace(-1,1,npts);

% x1 vs x2 at x3 = 0
[x1g,x2g] = meshgrid(v,v);
Z12 = f2(b_xan2, x1g, x2g, 0);

% Plot Surface & Contour for x1 vs x2
subplot(3,2,1);
surf(x1g, x2g, Z12, 'EdgeColor','none');hold on;
scatter3(Xc(:,1),Xc(:,2),Y(:,2),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Time');
zlabel('Predicted Betaxanthin (mg/100g)');
title('Betaxanthin Surface: Temp vs Time at Mass=1g');
colorbar; view(45,30);

subplot(3,2,2);
contour(x1g, x2g, Z12, 10, 'ShowText','on','LineWidth',1.2,'LabelFormat','%.1f');hold on;
scatter(Xc(:,1),Xc(:,2),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Time');
title('Betaxanthin Contour: Temp vs Time at Mass=1g');



% x1 vs x3 at x2 = 0
[x1g,x3g] = meshgrid(v,v);
Z13 = f2(b_xan2, x1g, 0, x3g);

% Plot Surface & Contour for x1 vs x3
subplot(3,2,3);
surf(x1g, x3g, Z13, 'EdgeColor','none');hold on;
scatter3(Xc(:,1),Xc(:,3),Y(:,2),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Mass');
zlabel('Predicted Betaxanthin (mg/100g)');
title('Betaxanthin Surface: Temp vs Mass at Time=70min');
colorbar; view(45,30);

subplot(3,2,4);
contour(x1g, x3g, Z13, 10, 'ShowText','on','LineWidth',1.2,'LabelFormat','%.1f');hold on;
scatter(Xc(:,1),Xc(:,3),50,'r','filled','MarkerEdgeColor','r');
xlabel('Temp');
ylabel('Time');
title('Betaxanthin Contour: Temp vs Mass at Time=70min');


% x2 vs x3 at x1 = 0
[x2g2,x3g2] = meshgrid(v,v);
Z23 = f2(b_xan2, 0, x2g2, x3g2);
% Plot Surface & Contour for x2 vs x3
subplot(3,2,5);
surf(x2g2, x3g2, Z23, 'EdgeColor','none');hold on;
scatter3(Xc(:,2),Xc(:,3),Y(:,2),50,'r','filled','MarkerEdgeColor','r');
xlabel('Time');
ylabel('Mass');
zlabel('Predicted Betaxanthin (mg/100g)');
title('Betaxanthin Surface: Time vs Mass at Temp=40degC');
colorbar; view(45,30);

subplot(3,2,6);
contour(x2g2, x3g2, Z23, 10, 'ShowText','on','LineWidth',1.2,'LabelFormat','%.1f');hold on;
scatter(Xc(:,2),Xc(:,3),50,'r','filled','MarkerEdgeColor','r');
xlabel('Time');
ylabel('Mass');
title('Betaxanthin Contour: Time vs Mass at Temp=40degC');


%% Question 5 - Optimization

%% Optimum for Betacyanin


f_rsm = @(b,x1,x2,x3) ...
     b(1) ...                
   + b(2).* x2  ...          
   + b(3).* x3  ...
   + b(4).* x2.*x3 ...
   + b(5).* x1.^2 ...
   + b(6).* x2.^2 ...
   + b(7).* x3.^2;

fun   = @(x) -f_rsm(b_cyan2, x(1),x(2),x(3));
x0    = [0 0 0];                      % start at center
lb    = [-1 -1 -1]; 
ub    = [ 1  1  1];
opts  = optimoptions('fmincon','Display','iter');
optimum_cyan  = fmincon(fun,x0,[],[],[],[],lb,ub,[],opts);

disp('The optimal coded point is:')
optimum_cyan
disp('Predicted betacyanin at optimum:')
f_rsm(b_cyan2, optimum_cyan(1), optimum_cyan(2), optimum_cyan(3))

xc = optimum_cyan;

% define spans
span.Temp =  50 - 30;   
span.Time = 120 -  20;   
span.Mass =  1.5- 0.5;    

% compute real values
T_real = 30 + (xc(1)+1)/2 * span.Temp;   % same as 40 + xc(1)*10
t_real = 20 + (xc(2)+1)/2 * span.Time;   % same as 70 + xc(2)*50
m_real = 0.5 + (xc(3)+1)/2 * span.Mass;  % same as 1  + xc(3)*0.5

fprintf('Real optimum: T=%.1f°C, Time=%.1f min, Mass=%.2f g\n', ...
         T_real, t_real, m_real);

npts = 61;
v    = linspace(-1,1,npts);

figure('Color','w','Units','normalized','Position',[.2 .4 .6 .3]);

% 1) Temp vs Time @ Mass = optimum_cyan(3)
subplot(1,3,1)
[X1,X2] = meshgrid(v,v);
Z12     = f_rsm(b_cyan2, X1, X2,  optimum_cyan(3));
contour(v,v,Z12,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_cyan(1), optimum_cyan(2),'ro','MarkerFaceColor','r')
hold off
xlabel('x_1 (Temp)'),  ylabel('x_2 (Time)')
title(sprintf('Temp vs Time @ Mass=%.2f',optimum_cyan(3)))

% 2) Temp vs Mass @ Time = optimum_cyan(2)
subplot(1,3,2)
[X1,X3] = meshgrid(v,v);
Z13     = f_rsm(b_cyan2, X1, optimum_cyan(2), X3);
contour(v,v,Z13,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_cyan(1), optimum_cyan(3),'ro','MarkerFaceColor','r')
hold off
xlabel('x_1 (Temp)'),  ylabel('x_3 (Mass)')
title(sprintf('Temp vs Mass @ Time=%.2f',optimum_cyan(2)))

% 3) Time vs Mass @ Temp = optimum_cyan(1)
subplot(1,3,3)
[X2,X3] = meshgrid(v,v);
Z23     = f_rsm(b_cyan2, optimum_cyan(1), X2, X3);
contour(v,v,Z23,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_cyan(2), optimum_cyan(3),'ro','MarkerFaceColor','r')
hold off
xlabel('x_2 (Time)'),  ylabel('x_3 (Mass)')
title(sprintf('Time vs Mass @ Temp=%.2f',optimum_cyan(1)))

% after plotting all three subplots:
h = colorbar('Position',[.91 .11 .02 .815]);  
h.Label.String = 'Predicted Betacyanin';

%% Optimum for Betaxanthin

f_xan = @(b,x1,x2,x3) ...
     b(1) ...                % intercept
   + b(2).* x1  ...          % x1
   + b(3).* x2  ...          % x2
   + b(4).* x3  ...          % x3
   + b(5).* x1.*x2 ...       % x1*x2
   + b(6).* x2.*x3 ...       % x2*x3
   + b(7).* x1.^2  ...       % x1^2
   + b(8).* x2.^2  ...       % x2^2
   + b(9).* x3.^2;           % x3^2

% --- 2) If you haven’t yet found the coded optimum for betaxanthin:
fun2   = @(x) -f_xan(b_xan2, x(1),x(2),x(3));
opts  = optimoptions('fmincon','Display','iter');
optimum_xan = fmincon(fun2, [0 0 0], [],[],[],[],[-1 -1 -1],[1 1 1],[],opts);

disp('The optimal coded point is:')
optimum_xan
disp('Predicted betaxanthin at optimum:')
f_xan(b_xan2, optimum_xan(1), optimum_xan(2), optimum_xan(3))

xc= optimum_xan;

% define spans
span.Temp =  50 - 30;   
span.Time = 120 -  20;   
span.Mass =  1.5- 0.5;    

% compute real values
T_real = 30 + (xc(1)+1)/2 * span.Temp;   
t_real = 20 + (xc(2)+1)/2 * span.Time;   
m_real = 0.5 + (xc(3)+1)/2 * span.Mass;

fprintf('Real optimum: T=%.1f°C, Time=%.1f min, Mass=%.2f g\n', ...
         T_real, t_real, m_real);



% --- 3) Now produce the 1×3 contour figure
npts = 61;
v    = linspace(-1,1,npts);

figure('Color','w','Units','normalized','Position',[.2 .4 .6 .3]);

% (a) Temp vs Time @ Mass = optimum_xan(3)
subplot(1,3,1);
[X1,X2] = meshgrid(v,v);
Z12     = f_xan(b_xan2, X1, X2,  optimum_xan(3));
contour(v,v,Z12,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_xan(1), optimum_xan(2), 'ro','MarkerFaceColor','r');
hold off
xlabel('x_1 (Temp)'); ylabel('x_2 (Time)');
title(sprintf('Temp vs Time @ Mass=%.2f',optimum_xan(3)));

% (b) Temp vs Mass @ Time = optimum_xan(2)
subplot(1,3,2);
[X1,X3] = meshgrid(v,v);
Z13     = f_xan(b_xan2, X1, optimum_xan(2), X3);
contour(v,v,Z13,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_xan(1), optimum_xan(3), 'ro','MarkerFaceColor','r');
hold off
xlabel('x_1 (Temp)'); ylabel('x_3 (Mass)');
title(sprintf('Temp vs Mass @ Time=%.2f',optimum_xan(2)));

% (c) Time vs Mass @ Temp = optimum_xan(1)
subplot(1,3,3);
[X2,X3] = meshgrid(v,v);
Z23     = f_xan(b_xan2, optimum_xan(1), X2, X3);
contour(v,v,Z23,12,'ShowText','on','LabelFormat','%.2f','LineWidth',1.2);
hold on
plot(optimum_xan(2), optimum_xan(3), 'ro','MarkerFaceColor','r');
hold off
xlabel('x_2 (Time)'); ylabel('x_3 (Mass)');
title(sprintf('Time vs Mass @ Temp=%.2f',optimum_xan(1)));

% --- 4) add a shared colorbar on the right
cb = colorbar('Position',[.91 .11 .02 .815]);  
cb.Label.String = 'Predicted Betaxanthin';
