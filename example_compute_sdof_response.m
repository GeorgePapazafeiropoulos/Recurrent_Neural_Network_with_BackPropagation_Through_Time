%% Local function: example_compute_sdof_response
function response = example_compute_sdof_response(force, dt, m, k, c)
% Compute SDOF response with dynamic time integration step by step
% procedure using Newmark-beta method 
n = length(force);
response = zeros(n, 1);
velocity = zeros(n, 1);
acceleration = zeros(n, 1);
% Initial conditions
response(1) = 0;
velocity(1) = 0;
acceleration(1) = (force(1) - c*velocity(1) - k*response(1)) / m;
% Newmark-beta parameters (constant average acceleration)
gamma = 0.5;
beta = 0.25;
% Effective stiffness
keff = k + (gamma/(beta*dt))*c + (1/(beta*dt^2))*m;
% Time integration
for i = 2:n
    % Effective force
    Feff = force(i) + ...
        m*(1/(beta*dt^2)*response(i-1) + 1/(beta*dt)*velocity(i-1) + (1/(2*beta)-1)*acceleration(i-1)) + ...
        c*(gamma/(beta*dt)*response(i-1) + (gamma/beta-1)*velocity(i-1) + dt*(gamma/(2*beta)-1)*acceleration(i-1));
    % Solve for displacement
    response(i) = Feff / keff;
    % Update velocity and acceleration
    acceleration(i) = (1/(beta*dt^2))*(response(i)-response(i-1)) - ...
        (1/(beta*dt))*velocity(i-1) - ((1/(2*beta))-1)*acceleration(i-1);
    velocity(i) = velocity(i-1) + dt*((1-gamma)*acceleration(i-1) + gamma*acceleration(i));
end
end

