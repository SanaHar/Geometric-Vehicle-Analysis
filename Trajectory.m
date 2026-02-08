clear; clc; close all;

%%  USER PARAMETERS 

videoFile = 'ConstantCurve.mp4';   % Path to the video
threshold_deg = 2;         % Angular threshold for orthogonality check (degrees)

% Camera calibration matrix (example, replace with your own)
K = [1855  0    574;
       0 1850   965;
       0   0     1];

car_width=1.8;
car_length=4.1;
d_lights=1.294;   % distance between rear lights [m]

%%  LOAD VIDEO 

v = VideoReader(videoFile);
nFrames = floor(v.Duration * v.FrameRate);

fprintf('Video loaded: %d frames available\n', nFrames);

%%  SELECT TWO FRAMES 

frameIdx1=140; %1
frameIdx2=200; %100

frame1=read(v,frameIdx1);
frame2=read(v,frameIdx2);

%%  FRAME 1 (L1, R1) 

figure;
imshow(frame1); hold on
title('L1 and R1 on frame 1')

L1 =[1160; 289; 1]; %[436; 236; 1];
R1 =[1649; 283; 1]; %[1452; 256; 1];

plot(L1(1), L1(2), 'g.', 'MarkerSize', 30, 'DisplayName', 'L1');
plot(R1(1), R1(2), 'c.', 'MarkerSize', 30, 'DisplayName', 'R1');
legend show;
hold off

%%  FRAME 2 (L2, R2) 

figure;
imshow(frame2); hold on
title('L2 and R2 on frame 2')

L2=[1306; 313; 1]; %[540; 272; 1];
R2=[1620; 314; 1]; %[1353; 291; 1];

plot(L2(1), L2(2), 'g.', 'MarkerSize', 30, 'DisplayName', 'L2');
plot(R2(1), R2(2), 'c.', 'MarkerSize', 30, 'DisplayName', 'R2');
legend show;
hold off;

%%  COMPUTE IMAGE LINES 

% Lines are computed as cross products of points
l_L1R1 = cross(L1, R1);
l_L2R2 = cross(L2, R2);

l_L1L2 = cross(L1, L2);
l_R1R2 = cross(R1, R2);

%%  COMPUTE VANISHING POINTS 

vx = cross(l_L1R1, l_L2R2);
vy = cross(l_L1L2, l_R1R2);

% Normalize homogeneous coordinates
vx = vx / vx(3);
vy = vy / vy(3);

fprintf('Vanishing point vx = [%f %f %f]\n', vx);
fprintf('Vanishing point vy = [%f %f %f]\n', vy);

%%  ORTHOGONALITY CHECK USING K 

% Convert vanishing points to direction vectors in camera frame
d_x=inv(K)*vx;
d_y=inv(K)*vy;

% Normalize directions
d_x=d_x/norm(d_x);
d_y=d_y/norm(d_y);

% Compute angle between directions
angle_rad = acos(dot(d_x, d_y));
angle_deg = rad2deg(angle_rad);

fprintf('Angle between directions = %.2f degrees\n', angle_deg);

%%  LINE AT INFINITY 

linf = cross(vx, vy);
linf = linf / norm(linf(1:2));

fprintf('Line at infinity linf = [%f %f %f]\n', linf);

%%  PLANE AT INFINITY (BACKPROJECTION) 

% Plane backprojection: pi = K^(-T) * linf
pi=K\linf;

% Normalize plane
pi=pi/norm(pi(1:3));

fprintf('Plane pi = [%f %f %f]\n', pi);

%% 3D SPACE POSITION

ray_L1=K\L1; ray_L1=ray_L1/norm(ray_L1); 
ray_R1=K\R1; ray_R1=ray_R1/norm(ray_R1); 

lambda1=d_lights/norm(ray_R1-ray_L1); 
pos_L1=lambda1*ray_L1; pos_R1=lambda1*ray_R1; 

ray_L2=K\L2; ray_L2=ray_L2/norm(ray_L2); 
ray_R2=K\R2; ray_R2=ray_R2/norm(ray_R2); 

lambda2=d_lights/norm(ray_R2-ray_L2); 
pos_L2=lambda2*ray_L2; pos_R2=lambda2*ray_R2;

mid1=(pos_L1+pos_R1)/2;
mid2=(pos_L2+pos_R2)/2;

% Correction on the second frame

vec_L1R1=pos_R1([1,3])-pos_L1([1,3]);
%yaw0=atan2(vec_L1R1(2), vec_L1R1(1));
%rad2deg(yaw0)

yaw0=deg2rad(-45);
yaw1=yaw0+deg2rad(90-angle_deg);
rad2deg(yaw1)

%new_pos_L1=pos_L1;
pos_L1(1)=mid1(1)-d_lights/2*cos(yaw0);
pos_L1(3)=mid1(3)-d_lights/2*sin(yaw0);

pos_R1(1)=mid1(1)+d_lights/2*cos(yaw0);
pos_R1(3)=mid1(3)+d_lights/2*sin(yaw0);

new_pos_L2=pos_L2;
new_pos_L2(1)=mid2(1)-d_lights/2*cos(yaw1);
new_pos_L2(3)=mid2(3)-d_lights/2*sin(yaw1);

new_pos_R2=pos_R2;
new_pos_R2(1)=mid2(1)+d_lights/2*cos(yaw1);
new_pos_R2(3)=mid2(3)+d_lights/2*sin(yaw1);

%% PLOT

figure; hold on;
plot3(pos_L1(1), pos_L1(2), pos_L1(3), 'ro','MarkerSize', 8, 'LineWidth', 2, ...
    'DisplayName','L1');
plot3(pos_R1(1), pos_R1(2), pos_R1(3), 'ro','MarkerSize', 8, 'LineWidth', 2, ...
     'DisplayName','R1');
plot3([pos_L1(1), pos_R1(1)], [pos_L1(2), pos_R1(2)], [pos_L1(3), pos_R1(3)], ...
    'r--', 'LineWidth', 1,'DisplayName','Virtual line between L1 and R1');

plot3(new_pos_L2(1), new_pos_L2(2), new_pos_L2(3), 'ro','MarkerSize', 8, 'LineWidth', 2, ...
     'DisplayName','L2');
plot3(new_pos_R2(1), new_pos_R2(2), new_pos_R2(3), 'ro','MarkerSize', 8, 'LineWidth', 2, ...
     'DisplayName','R2');
plot3([new_pos_L2(1), new_pos_R2(1)], [new_pos_L2(2), new_pos_R2(2)], [new_pos_L2(3), new_pos_R2(3)], ...
    'r--', 'LineWidth', 1,'DisplayName','Virtual line between L2 and R2');

plot3(mid1(1), mid1(2), mid1(3), 'k.','MarkerSize', 20, 'LineWidth', 2, ...
    'DisplayName','Midpoint1');
plot3(mid2(1), mid2(2), mid2(3), 'k.','MarkerSize', 20, 'LineWidth', 2, ...
    'DisplayName','Midpoint2');

disp('Frame 1 car coordinates wrt camera (x,y,z):'); disp(mid1);
disp('Frame 2 car coordinates wrt camera (x,y,z):'); disp(mid2);

 grid on; axis equal;
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title('Lights 3D position and car trajectory (coordinates wrt the camera');
    legend show;

if abs(angle_deg - 90) < threshold_deg
    disp('The directions are orthogonal (within the threshold), the car is moving forward.');

    plot3([mid1(1), mid2(1)], [mid1(2), mid2(2)], [mid1(3), mid2(3)], ...
    'k--', 'LineWidth', 2,'DisplayName','Car trajectory');
else
    disp('The directions are not orthogonal (within the threshold), the car in steering.');
    
    %%  CURVATURE ESTIMATION (using I.C.R. from rear lights) 
    
    figure; hold on
    title('Car road plane position and trajectory (coordinates wrt the camera) / Steering case');
    
    dra=0.7; % Distance between rear axle and lights

    C1=[mid1(1)-car_length/2*sin(yaw0),mid1(3)+car_length/2*cos(yaw0)];
    C2=[mid2(1)-car_length/2*sin(yaw1),mid2(3)+car_length/2*cos(yaw1)];
    
    C1ra=[mid1(1)-dra*sin(yaw0),mid1(3)+dra*cos(yaw0)];
    C2ra=[mid2(1)-dra*sin(yaw1),mid2(3)+dra*cos(yaw1)];

    C1ral=[C1ra(1)-car_width/2*cos(yaw0),C1ra(2)-car_width/2*sin(yaw0)];
    C2ral=[C2ra(1)-car_width/2*cos(yaw1),C2ra(2)-car_width/2*sin(yaw1)];

    plot(C1(1),C1(2),'k.','MarkerSize',20,'DisplayName','Car center 1');
    plot(C2(1),C2(2),'k.','MarkerSize',20,'DisplayName','Car center 2');

    plot(C1ra(1),C1ra(2),'g.','MarkerSize',25,'DisplayName','Rear axle center 1');
    plot(C2ra(1),C2ra(2),'g.','MarkerSize',25,'DisplayName','Rear axle center 2');
    
    plot(C1ral(1),C1ral(2),'b.','MarkerSize',15,'DisplayName','C1l');
    plot(C2ral(1),C2ral(2),'b.','MarkerSize',15,'DisplayName','C2l');

    m1=(C1ra(2)-C1ral(2))/(C1ra(1)-C1ral(1)); % First line equation
    q1=-(m1*C1ra(1)-C1ra(2));

    m2=(C2ra(2)-C2ral(2))/(C2ra(1)-C2ral(1)); % Second line equation
    q2=-(m2*C2ra(1)-C2ra(2));

    x_icr=(q2-q1)/(m1-m2);
    z_icr=(m1*x_icr+q1);

    plot(pos_L1(1), pos_L1(3), 'ro','MarkerSize', 8, 'LineWidth', 2, 'DisplayName','L1');
    plot(pos_R1(1), pos_R1(3), 'ro','MarkerSize', 8, 'LineWidth', 2, 'DisplayName','R1');
    plot([pos_L1(1), pos_R1(1)], [pos_L1(3), pos_R1(3)], ...
    'r--', 'LineWidth', 1,'DisplayName','Virtual line between L1 and R1');
   
    plot(new_pos_L2(1), new_pos_L2(3), 'ro','MarkerSize', 8, 'LineWidth', 2, 'DisplayName','L2');
    plot(new_pos_R2(1), new_pos_R2(3), 'ro','MarkerSize', 8, 'LineWidth', 2, 'DisplayName','R2');
    plot([new_pos_L2(1), new_pos_R2(1)], [new_pos_L2(3), new_pos_R2(3)], ...
    'r--', 'LineWidth', 1,'DisplayName','Virtual line between L2 and R2');
    
    plot([x_icr, C1ra(1)], [z_icr, C1ra(2)], ...
    'k--', 'LineWidth', 1,'DisplayName','Virtual radius 1');
    plot([x_icr, C2ra(1)], [z_icr, C2ra(2)], ...
    'k--', 'LineWidth', 1,'DisplayName','Virtual radius 2');

    drawRotatedRectangle(C1, car_length, car_width, 3.14/2+yaw0);
    drawRotatedRectangle(C2, car_length, car_width, 3.14/2+yaw1);

    plot(x_icr, z_icr, 'mx','MarkerSize',12,'LineWidth',3,'DisplayName','ICR');
    xlabel('X [m]'); ylabel('Z [m]')

    R1=sqrt((mid1(1)-x_icr)^2+(mid1(3)-z_icr)^2);
    R2=sqrt((mid2(1)-x_icr)^2+(mid2(3)-z_icr)^2);
    R=(R1+R2)/2;

    fprintf('The radius of the curve is approximately: %.1f m \n',R);
    
    theta = linspace(0, 2*3.14, 100);
    x_circle=x_icr+R*cos(theta); z_circle=z_icr+R*sin(theta);

    plot(x_circle, z_circle, 'm--', 'LineWidth',1, 'DisplayName','Trajectory circle');
    
    grid on; axis equal; legend show
end

function drawRotatedRectangle(C1, L, W, theta)

    % Coordinates
    x = [-L/2,  L/2,  L/2, -L/2, -L/2];
    y = [-W/2, -W/2,  W/2,  W/2, -W/2];

    % Rotation matrix
    R = [cos(theta), -sin(theta);
         sin(theta),  cos(theta)];

    % Rotation
    rotated = R * [x; y];
    
    %Translation
    x_rot = rotated(1,:) + C1(1);
    y_rot = rotated(2,:) + C1(2);

    % Draw
    plot(x_rot, y_rot, 'k-', 'LineWidth', 1,'DisplayName',"Vehicle's space occupancy");
    axis equal
    grid on
    hold on
end




