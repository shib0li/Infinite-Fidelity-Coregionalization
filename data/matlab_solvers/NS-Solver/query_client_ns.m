function query_client_ns(X, m, buffer)


% addpath_recurse('./sobol')

grid_num = m-1;

n_run = size(X,1); dim_run = size(X, 2);
Re_downlimit=10;
Re_uplimit=500;
X(:,1)=X(:,1)*(Re_uplimit-Re_downlimit)+ones(n_run,1)*Re_downlimit;

Lid_uplimit=10;
Lid_downlimit=1;
X(:,2:5)=X(:,2:5)*(Lid_uplimit-Lid_downlimit)+ones(n_run,dim_run-1)*Lid_downlimit;

%% model setting
% Re = 5e2;     % Reynolds number
dt = 1e-3;     % time step
tf = 10;    % final time
lx = 1;       % width of box
ly = 1;       % height of box
nx = grid_num;      % number of x-gridpoints
ny = grid_num;      % number of y-gridpoints
%nsteps = 20;  % number of steps with graphic output %20
nsteps=20;

%% run
resultName = buffer;

URec=[];
VRec=[];
PRec=[];
% save(resultName,'-v7.3');
save(resultName,'-v7.3','-nocompression')
%returned mat, field X is input, field URec, VRec & PRec are outputs, the
%last dimension is the index
for i = 1:n_run
    t_start = tic;
    disp(i)
    Re=X(i,1);
    % 5 parameters
    BC.uN=X(i,2);     BC.vN=0;
    BC.uS=X(i,3);     BC.vS=0;
    BC.uW=0;          BC.vW=X(i,4);
    BC.uE=0;          BC.vE=X(i,5);      
    [Ut,Vt,Pt]=  mit_ns_t_v4(Re,dt,tf,lx,ly,nx,ny,nsteps,BC,true);

    %pause(2)
    m = matfile(resultName,'Writable',true);
    m.URec = cat(4, m.URec, Ut);
    m.VRec = cat(4, m.VRec, Vt);
    m.PRec = cat(4, m.PRec, Pt);
    m=[];
    
    t_end = toc(t_start);
    fprintf('%d-th sample with gridsize=%d takes %.5f sec\n', i, grid_num, t_end);
end

pause(5)

end

