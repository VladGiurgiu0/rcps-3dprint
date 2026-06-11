%% =========================== RCPS_v4.m =====================================
% like RCPS_v3.m but with the possiblity to keep sides of the tile without cutting it.
% Load packing.xyzd -> build scalar field -> mesh -> STL
%
% Python deps (install into your MAIN python env; not the project folder):
%   python3 -m pip install -U pip
%   python3 -m pip install -U numpy meshio trimesh
%   python3 -m pip install -U pyvista vtk
%   python3 -m pip install -U scikit-image PyMCubes
%   python3 -m pip install -U iso2mesh     % pyiso2mesh
%   python3 -m pip install -U pymeshfix
%   python3 -m pip install -U lxml networkx
% -------------------------------------------------------------------------

clear; clc; close all
set(0,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% =========================== USER INPUTS ================================
p = struct();

% ------------------------------ Paths -----------------------------------
p.paths.root     = '/Users/vlad/Claude/Morphos/RCPS for 3D printing/data_example/';   % folder with packing.xyzd
p.paths.packing  = fullfile(p.paths.root,'packing.xyzd');
p.paths.outDir   = '/Users/vlad/Claude/Morphos/RCPS for 3D printing/tests/fixtures/';

% --------------------------- Domain [mm] --------------------------------
% tile MUST match the packing generator domain
p.geom.L_tile = 50;  p.geom.H_tile = 50;  p.geom.W_tile = 50;
% facility is the final printable domain - recommended: the same size as
% tile and 3D print multiple tiles so high domain resolution can be
% achieved
p.geom.L_fac  = 50;  p.geom.H_fac  = 50;  p.geom.W_fac  = 50;

p.geom.phi = 0.35;  % phi = porosity = volume_empty / (volume_empty + volume_spheres);

% -------------------------- Sphere options ------------------------------
p.spheres.diameter = 6; % [mm]

% --------------------- Field generation (voxel/icsg) --------------------
p.field.mode       = 'icsg';       % 'icsg' (recommended) | 'voxel'
p.field.exportWhat = 'beads';      % 'beads' (recommended) | 'pore'
p.field.exportMode = 'tile';   % 'tile'  | 'facility' (recommended)
p.field.ghostTiles = 1;            % keep boundary spheres (facility tiling) (1 - recommended)
p.field.padVox     = 1;            % pad scalar field to extract clean box walls (1 - recommended)
p.field.bandVox    = 3;            % only for 'icsg' (band-limited union near spheres) (3 - recommended)
p.field.keepSides  = {};           % e.g. {'R','L'} or {'R','L','A','P','S','I'}; RAS faces where beads are NOT cut

% ----------------------------- Grid -------------------------------------
p.grid.voxSize_mm  = 0.1;

% ------------------------- 3D-printability ------------------------------
p.spheres.expansion_factor = 1.00;   % print diameter multiplier (1 - recommended)
p.spheres.contactTol_mm    = 0.20;   % used only if bridges are enabled (0.2 - recommended)
p.bridge.mode        = 'cylinders'; % 'none' | 'cylinders' (recommended) | 'cylinders+diameter' 
p.bridge.radiusFrac  = 0.15;        % cyl radius = radiusFrac * min(r_nom_i,r_nom_j) (0.15 - recommended)

% -------------------------- Meshing backend -----------------------------
p.mesh.backend = 'matlab';          % 'matlab' | 'python'

% mesh method (depends on backend)
% MATLAB: 'extractisosurface' | 'isosurface' | 'iso2mesh' (recommended)
% Python: 'flying_edges' | 'marching_cubes' | 'contour' | 'skimage' | 'pymcubes' | 'iso2mesh' (pyiso2mesh)
p.mesh.method  = 'iso2mesh';

% ----------------------- iso2mesh quality knobs -------------------------
% Used when p.mesh.method == 'iso2mesh' (MATLAB iso2mesh OR Python pyiso2mesh)
% Interpreted in voxel/grid units unless you convert explicitly later.
p.mesh.iso2mesh.angbound_deg = 25;      % min angle constraint (deg)
p.mesh.iso2mesh.radbound     = 1.0;     % max Delaunay circle radius (triangle size) (1 - recommended)
p.mesh.iso2mesh.distbound    = 0.10;    % max Delaunay sphere distance (surface error) (0.1 - recommended)
p.mesh.iso2mesh.maxnode      = 2e8;     % hard cap on surface nodes

% ------------- Mesh post-processing (for matlab backend)------------------
p.mesh.doRepair      = false;
p.mesh.doReducePatch = false;
p.mesh.reduceFactor  = 0.25;

% ------------------------- Python runner --------------------------------
% used ONLY if p.mesh.backend='python'
p.py.exe = '/Users/vlad/miniforge3/bin/python3'; % python3 exe location                    
p.py.script  = fullfile('mesh_from_raw.py'); % name of the python file to be ran

% ---------------------------- Output ------------------------------------
%p.out.baseName = sprintf('packed_%s_%s_%s_%s_vox_%g',p.field.mode, p.field.exportWhat, p.mesh.backend, p.mesh.method, p.grid.voxSize_mm);
p.out.baseName = 'reference';
p.out.stl      = fullfile(p.paths.outDir, [p.out.baseName '.stl']);
p.out.threemf     = fullfile(p.paths.outDir, [p.out.baseName '.3mf']);
p.out.writeInfoFile = true;
% only works for matlab backend. python exports .3mf only
p.out.saveSTL   = false;   % if false: do not write .stl (false - recommended)
p.out.save3MF   = true;   % if false: do not write .3mf (true - recommended)

% --- show figures
p.show.figures = false; % (false - recommended)

% --- encode keepSides into output basename (e.g. _keep_RLAPSI)
if isfield(p,'field') && isfield(p.field,'keepSides') && ~isempty(p.field.keepSides)
    keepStr = upper(strjoin(p.field.keepSides,''));
    p.out.baseName = [p.out.baseName '_keep_' keepStr];
    p.out.stl      = fullfile(p.paths.outDir, [p.out.baseName '.stl']);
    p.out.threemf   = fullfile(p.paths.outDir, [p.out.baseName '.3mf']);
end

%% time in
tic
%% ========================= STEP 0: INITIAL GUESS =========================
% this section is needed so I can start the packing-generation with the right parameters
% Calculate the volume of the box and the volume of a single sphere
boxVolume = p.geom.L_tile * p.geom.H_tile * p.geom.W_tile;               % Volume of the box [mm^3]
sphereVolume = (4/3) * pi * (p.spheres.diameter/2)^3; % Volume of a single sphere [mm^3]

% Calculate the effective volume occupied by the spheres based on porosity
effectiveVolume = boxVolume * (1 - p.geom.phi); % Volume occupied by spheres [mm^3]

% Calculate the number of spheres that can fit in the effective volume
numSpheres = floor(effectiveVolume / sphereVolume);

% Display
fprintf('Box dimensions: L = %.0f mm, H = %.0f mm, W = %.0f mm\n', p.geom.L_tile, p.geom.H_tile, p.geom.W_tile);
fprintf('Bead diameter: d = %.3f mm\n', p.spheres.diameter);
fprintf('Porosity: phi = %.3f\n\n', p.geom.phi);

fprintf('Box volume: %.3e mm^3\n', boxVolume);
fprintf('Single sphere volume: %.3e mm^3\n', sphereVolume);
fprintf('Effective volume for spheres (1-phi): %.3e mm^3\n\n', effectiveVolume);

fprintf('Estimated number of spheres that can be packed: %d spheres\n', numSpheres);


%% ========================= STEP 1: LOAD PACKING =========================
fprintf('Loading packing: %s\n', p.paths.packing);
fid = fopen(p.paths.packing,'r','ieee-le');
assert(fid>0,'Cannot open file: %s', p.paths.packing);
raw = fread(fid, Inf, 'double=>double');
fclose(fid);
assert(mod(numel(raw),4)==0,'Invalid xyzd stream (count not divisible by 4).');

S  = reshape(raw,4,[]).';     % [x y z d]
x0 = S(:,1); y0 = S(:,2); z0 = S(:,3); d0 = S(:,4);
N0 = size(S,1);
fprintf('Loaded spheres: N = %d\n', N0);

%% ================= STEP 2: CHOOSE DOMAIN + REPLICATE CENTERS =============
switch lower(p.field.exportMode)
    case 'tile'
        dom.L = p.geom.L_tile; dom.H = p.geom.H_tile; dom.W = p.geom.W_tile;
        x = x0; y = y0; z = z0;
        d = d0;

        dom.nTx = 1; dom.nTy = 1; dom.nTz = 1;

    case 'facility'
        dom.L = p.geom.L_fac; dom.H = p.geom.H_fac; dom.W = p.geom.W_fac;

        dom.nTx = floor(p.geom.L_fac / p.geom.L_tile);
        dom.nTy = floor(p.geom.H_fac / p.geom.H_tile);
        dom.nTz = floor(p.geom.W_fac / p.geom.W_tile);
        assert(dom.nTx>=1 && dom.nTy>=1 && dom.nTz>=1, 'Facility too small to fit at least one tile.');

        txList = (-p.field.ghostTiles):(dom.nTx-1+p.field.ghostTiles);
        tyList = (-p.field.ghostTiles):(dom.nTy-1+p.field.ghostTiles);
        tzList = (-p.field.ghostTiles):(dom.nTz-1+p.field.ghostTiles);
        nCopies = numel(txList)*numel(tyList)*numel(tzList);

        x = repmat(x0, nCopies, 1);
        y = repmat(y0, nCopies, 1);
        z = repmat(z0, nCopies, 1);
        d = repmat(d0, nCopies, 1);

        cnt = 0;
        for tz = tzList
            for ty = tyList
                for tx = txList
                    idx = (1:N0) + cnt*N0;
                    x(idx) = x(idx) + tx*p.geom.L_tile;
                    y(idx) = y(idx) + ty*p.geom.H_tile;
                    z(idx) = z(idx) + tz*p.geom.W_tile;
                    cnt = cnt + 1;
                end
            end
        end

    otherwise
        error('p.field.exportMode must be ''tile'' or ''facility''.');
end

% expansion (nominal + print)
r_nom = 0.5 * d;                           % nominal radius from file
r     = r_nom * p.spheres.expansion_factor; % print radius
d     = 2*r;                                % print diameter

Ns = numel(r);
fprintf('Domain: L=%.3g H=%.3g W=%.3g mm | tiles: %d %d %d | spheres: %d\n', ...
    dom.L, dom.H, dom.W, dom.nTx, dom.nTy, dom.nTz, Ns);
fprintf('Expansion factor: %.6f\n', p.spheres.expansion_factor);

%% ================== STEP 3: SNAP GRID (DOMAIN-DIVISIBLE) ================
voxSize = p.grid.voxSize_mm;

nx0 = max(8, round(dom.L/voxSize));
voxSize = dom.L / nx0;                     % snap so L is divisible

ny0 = round(dom.H/voxSize);
nz0 = round(dom.W/voxSize);

assert(abs(dom.H/voxSize - ny0) < 1e-10, 'H not divisible by voxSize after snap');
assert(abs(dom.W/voxSize - nz0) < 1e-10, 'W not divisible by voxSize after snap');

p.grid.voxSize_mm_snapped = voxSize;       % keep for naming + reproducibility
p.grid.nx0 = nx0; p.grid.ny0 = ny0; p.grid.nz0 = nz0;

fprintf('Grid (snapped): nx=%d ny=%d nz=%d | voxSize=%.6g mm\n', nx0, ny0, nz0, voxSize);

% overwrite output names to reflect snapped voxel size
%p.out.baseName = sprintf('packed_%s_%s_vox_%g', p.field.mode, p.mesh.method, p.grid.voxSize_mm_snapped);
%p.out.stl      = fullfile(p.paths.outDir, [p.out.baseName '.stl']);

%% ===================== STEP 4: BUILD SCALAR FIELD =======================
voxSize = p.grid.voxSize_mm_snapped;

% --- ADD: allow keeping boundary beads uncut on selected RAS faces (R/L/A/P/S/I)
padVox = p.field.padVox;
bandVoxEff = max(2, p.field.bandVox);
bandDist   = bandVoxEff * voxSize;

if isfield(p.field,'keepSides') && ~isempty(p.field.keepSides)
    rMax = max(r);
    padExtraVox = ceil((rMax + bandDist)/voxSize) + 1;
    padVox = padVox + padExtraVox;
end

nx = p.grid.nx0 + 2*padVox;
ny = p.grid.ny0 + 2*padVox;
nz = p.grid.nz0 + 2*padVox;

% --- ADD: feasibility guard for dense volumes ---
nVox = double(nx) * double(ny) * double(nz);
bytesF = 4 * nVox; % float32
if nVox > 5e8
    error(['Dense grid too large: nx*ny*nz = %.3g voxels (%.2f GiB for ONE float32 field). ', ...
           'For 200mm @ 0.025mm this is ~5.12e11 voxels (~1.86 TiB). ', ...
           'Use adaptive/sparse representations (e.g. OpenVDB volumeToMesh) or direct union-of-balls meshing (CGAL).'], ...
           nVox, bytesF/1024^3);
end

origin  = [-padVox*voxSize, -padVox*voxSize, -padVox*voxSize];

xVec = origin(1) + (0:nx-1)*voxSize;
yVec = origin(2) + (0:ny-1)*voxSize;
zVec = origin(3) + (0:nz-1)*voxSize;

% Box implicit (positive inside [0,L]x[0,H]x[0,W])
fx = min(xVec, dom.L - xVec);
fy = min(yVec, dom.H - yVec);
fz = min(zVec, dom.W - zVec);

% --- ADD: expand clipping box so selected faces do NOT cut beads (RAS convention)
xmin = 0; xmax = dom.L;
ymin = 0; ymax = dom.H;
zmin = 0; zmax = dom.W;

if isfield(p.field,'keepSides') && ~isempty(p.field.keepSides)
    % make the far-away face lie OUTSIDE the padded grid, so it never clips geometry
    rMax = max(r);
    extMm = 2*(rMax + bandDist) + 2*voxSize;
    ks = upper(string(p.field.keepSides));
    if any(ks=="L"), xmin = xmin - extMm; end
    if any(ks=="R"), xmax = xmax + extMm; end
    if any(ks=="P"), ymin = ymin - extMm; end
    if any(ks=="A"), ymax = ymax + extMm; end
    if any(ks=="I"), zmin = zmin - extMm; end
    if any(ks=="S"), zmax = zmax + extMm; end
end

cx = (xmin + xmax)/2;  bx = (xmax - xmin)/2;
cy = (ymin + ymax)/2;  by = (ymax - ymin)/2;
cz = (zmin + zmax)/2;  bz = (zmax - zmin)/2;
qx = abs(xVec - cx) - bx;
qy = abs(yVec - cy) - by;
qz = abs(zVec - cz) - bz;

QX = reshape(single(qx),[],1,1);
QY = reshape(single(qy),1,[],1);
QZ = reshape(single(qz),1,1,[]);

QXp = max(QX,0); QYp = max(QY,0); QZp = max(QZ,0);
outside = sqrt(QXp.^2 + QYp.^2 + QZp.^2);
inside  = min(max(max(QX,QY),QZ), 0);   % <=0 inside

sd_box = outside + inside;  % negative inside, positive outside
F_box  = sd_box;            % negative inside  (STRICT SDF)



% --- ADD: cull spheres that cannot influence the padded grid (major speedup for tiling) ---
xMinF = xVec(1);  xMaxF = xVec(end);
yMinF = yVec(1);  yMaxF = yVec(end);
zMinF = zVec(1);  zMaxF = zVec(end);

if strcmpi(p.field.mode,'icsg')
    radCull = r + bandDist;
else
    radCull = r;
end

keep = (x + radCull >= xMinF) & (x - radCull <= xMaxF) & ...
       (y + radCull >= yMinF) & (y - radCull <= yMaxF) & ...
       (z + radCull >= zMinF) & (z - radCull <= zMaxF);

Ns_before = Ns;
x = x(keep); y = y(keep); z = z(keep);
r = r(keep); d = d(keep);
r_nom = r_nom(keep);
Ns = numel(r);

fprintf('Culled spheres: %d -> %d kept (%.2f%% removed)\n', ...
    Ns_before, Ns, 100*(Ns_before - Ns)/max(1,Ns_before));

% --- ADD: if a face is kept uncut, drop ghost spheres beyond that face (avoid extra beads)
if isfield(p.field,'keepSides') && ~isempty(p.field.keepSides)
    ks = upper(string(p.field.keepSides));
    tol = 1e-12;
    keep2 = true(size(x));
    if any(ks=="L"), keep2 = keep2 & (x >= 0 - tol); end
    if any(ks=="R"), keep2 = keep2 & (x <= dom.L + tol); end
    if any(ks=="P"), keep2 = keep2 & (y >= 0 - tol); end
    if any(ks=="A"), keep2 = keep2 & (y <= dom.H + tol); end
    if any(ks=="I"), keep2 = keep2 & (z >= 0 - tol); end
    if any(ks=="S"), keep2 = keep2 & (z <= dom.W + tol); end

    Ns_before2 = Ns;
    x = x(keep2); y = y(keep2); z = z(keep2);
    r = r(keep2); d = d(keep2);
    r_nom = r_nom(keep2);
    Ns = numel(r);
    fprintf('Applied keepSides filter: %d -> %d kept\n', Ns_before2, Ns);
end




switch lower(p.field.mode)
    case 'icsg'

        % This preserves the full signed range [-bandDist, +R] inside the narrow band.
        F_beads = single(bandDist) * ones(nx,ny,nz,'single');     % preserves band
        % or, if you do NOT want truncation:
        % F_beads = -inf(nx,ny,nz,'single');  % keep true (non-SDF) implicit values


        t0 = tic;
        for i = 1:Ns
            xc = x(i); yc = y(i); zc = z(i); R = r(i);
            rad = R + bandDist;  % narrow-band evaluation radius

            ix1 = floor((xc - rad - origin(1))/voxSize) + 1;  ix2 = ceil((xc + rad - origin(1))/voxSize) + 1;
            iy1 = floor((yc - rad - origin(2))/voxSize) + 1;  iy2 = ceil((yc + rad - origin(2))/voxSize) + 1;
            iz1 = floor((zc - rad - origin(3))/voxSize) + 1;  iz2 = ceil((zc + rad - origin(3))/voxSize) + 1;

            ix1 = max(ix1,1); ix2 = min(ix2,nx);
            iy1 = max(iy1,1); iy2 = min(iy2,ny);
            iz1 = max(iz1,1); iz2 = min(iz2,nz);

            if (ix1>ix2) || (iy1>iy2) || (iz1>iz2), continue; end

            X = xVec(ix1:ix2) - xc;
            Y = yVec(iy1:iy2) - yc;
            Z = zVec(iz1:iz2) - zc;

            [XX,YY,ZZ] = ndgrid(single(X), single(Y), single(Z));
            local = sqrt(XX.^2 + YY.^2 + ZZ.^2) - single(R);

            blk = F_beads(ix1:ix2,iy1:iy2,iz1:iz2);
            F_beads(ix1:ix2,iy1:iy2,iz1:iz2) = min(blk, local);
        end
        fprintf('ICS G spheres done in %.2fs\n', toc(t0));

    case 'voxel'
        solid = false(nx,ny,nz);

        t0 = tic;
        for i = 1:Ns
            xc = x(i); yc = y(i); zc = z(i); R = r(i);
            rad = R;

            ix1 = floor((xc - rad - origin(1))/voxSize) + 1;  ix2 = ceil((xc + rad - origin(1))/voxSize) + 1;
            iy1 = floor((yc - rad - origin(2))/voxSize) + 1;  iy2 = ceil((yc + rad - origin(2))/voxSize) + 1;
            iz1 = floor((zc - rad - origin(3))/voxSize) + 1;  iz2 = ceil((zc + rad - origin(3))/voxSize) + 1;

            ix1 = max(ix1,1); ix2 = min(ix2,nx);
            iy1 = max(iy1,1); iy2 = min(iy2,ny);
            iz1 = max(iz1,1); iz2 = min(iz2,nz);

            if (ix1>ix2) || (iy1>iy2) || (iz1>iz2), continue; end

            X = xVec(ix1:ix2) - xc;
            Y = yVec(iy1:iy2) - yc;
            Z = zVec(iz1:iz2) - zc;

            [XX,YY,ZZ] = ndgrid(X, Y, Z);
            solid(ix1:ix2,iy1:iy2,iz1:iz2) = solid(ix1:ix2,iy1:iy2,iz1:iz2) | (XX.^2 + YY.^2 + ZZ.^2 <= R^2);
        end
        fprintf('VOX spheres done in %.2fs\n', toc(t0));

    otherwise
        error('p.field.mode invalid');
end

%% ===================== STEP 4b: OPTIONAL BRIDGES ========================
if ~strcmpi(p.bridge.mode,'none')
    assert(exist('rangesearch','file')==2, 'Bridge mode requires rangesearch (Statistics and Machine Learning Toolbox).');

    ctr = [x y z];
    searchR = 2*max(r) + p.spheres.contactTol_mm;
    nbrs = rangesearch(ctr, ctr, searchR);

    t0 = tic;
    for i = 1:Ns
        ni = nbrs{i};
        ni = ni(ni>i); % upper triangle

        c1 = ctr(i,:);
        for kk = 1:numel(ni)
            j = ni(kk);
            c2 = ctr(j,:);
            v  = c2 - c1;
            Lc = norm(v);
            if Lc<=0, continue; end

            if Lc > (r(i) + r(j) + p.spheres.contactTol_mm)
                continue
            end

            rcyl = p.bridge.radiusFrac * min(r(i), r(j));

            % local bounding box (add bandDist in ICSG so cylinder also band-limited)
            pad = rcyl + (strcmpi(p.field.mode,'icsg') * bandDist);

            xMin = min(c1(1),c2(1)) - pad;  xMax = max(c1(1),c2(1)) + pad;
            yMin = min(c1(2),c2(2)) - pad;  yMax = max(c1(2),c2(2)) + pad;
            zMin = min(c1(3),c2(3)) - pad;  zMax = max(c1(3),c2(3)) + pad;

            ix1 = floor((xMin - origin(1))/voxSize) + 1;  ix2 = ceil((xMax - origin(1))/voxSize) + 1;
            iy1 = floor((yMin - origin(2))/voxSize) + 1;  iy2 = ceil((yMax - origin(2))/voxSize) + 1;
            iz1 = floor((zMin - origin(3))/voxSize) + 1;  iz2 = ceil((zMax - origin(3))/voxSize) + 1;

            ix1 = max(ix1,1); ix2 = min(ix2,nx);
            iy1 = max(iy1,1); iy2 = min(iy2,ny);
            iz1 = max(iz1,1); iz2 = min(iz2,nz);

            if (ix1>ix2) || (iy1>iy2) || (iz1>iz2), continue; end

            X = xVec(ix1:ix2);
            Y = yVec(iy1:iy2);
            Z = zVec(iz1:iz2);
            [XX,YY,ZZ] = ndgrid(X, Y, Z);

            u = v / Lc;
            dx = XX - c1(1); dy = YY - c1(2); dz = ZZ - c1(3);
            t  = dx*u(1) + dy*u(2) + dz*u(3);
            t  = max(0, min(Lc, t));
            cx = c1(1) + t*u(1);
            cy = c1(2) + t*u(2);
            cz = c1(3) + t*u(3);

            dist = sqrt( (XX-cx).^2 + (YY-cy).^2 + (ZZ-cz).^2 );

            switch lower(p.field.mode)
                case 'icsg'
                    local = single(dist) - single(rcyl) - 0.25*voxSize;
                    blk   = F_beads(ix1:ix2,iy1:iy2,iz1:iz2);
                    F_beads(ix1:ix2,iy1:iy2,iz1:iz2) = min(blk, local);

                case 'voxel'
                    solid(ix1:ix2,iy1:iy2,iz1:iz2) = solid(ix1:ix2,iy1:iy2,iz1:iz2) | (dist <= rcyl);
            end
        end
    end
    fprintf('Bridges done in %.2fs\n', toc(t0));
end

%% ===================== STEP 4c: FINAL FIELD F ===========================
%isoLevel = 0;
isoLevel = -1e-6 * p.grid.voxSize_mm_snapped;

switch lower(p.field.mode)
    case 'icsg'
        % Confine to box; include cut beads at boundaries (if they intersect the box)
        F_beads = max(F_beads, F_box);

    % case 'voxel'
    %     % Convert binary to signed distance (positive inside solid)
    %     assert(exist('bwdist','file')==2, 'Voxel mode requires bwdist (Image Processing Toolbox).');
    %     dist_in  = bwdist(~solid);
    %     dist_out = bwdist( solid);
    %     F_beads  = single(dist_in - dist_out) * single(voxSize);
    %     F_beads  = min(F_beads, F_box);
end

switch lower(p.field.exportWhat)
    case 'beads'
        F = F_beads;

    case 'pore'
        % pore = inside box AND outside beads
        F = max(F_box,F_beads);

    otherwise
        error('p.field.exportWhat must be ''beads'' or ''pore''.');
end

fprintf('Field ready: size = [%d %d %d], isoLevel = %.3g\n', nx, ny, nz, isoLevel);

if p.show.figures
    figure(1); hold on; box on; grid on;
    fielddd = F(200:400,200:400,100);
    imagesc(fielddd); colormap(flipud(parula(16))); 
    contour(fielddd, [0 0], 'w', 'LineWidth', 1);
    %title('Colorbar: F, value of white iso-line: 0'); 
    axis image;
    clb = colorbar; clb.Label.String = '$F$'; clb.TickLabelInterpreter = 'latex'; clb.Label.Interpreter='latex';
    clb.FontSize=24;
    clim([-3 1])
    set(gca,'FontSize',24,'TickLabelInterpreter','latex')
    xlabel('$x \times 10$ [mm]')
    ylabel('$y \times 10$ [mm]')
    xlim([0 202])
    ylim([0 202])
    
end

fprintf('F_beads stats: min=%.6g (should be around -diameter/2) max=%.6g (should be arround bandDist) bandDist=%.6g\n', ...
min(F_beads,[],'all'), max(F_beads,[],'all'), bandDist);

%% ================= STEP 5a: EXPORT VOLUME FOR 3D SLICER ==================
switch p.field.mode
    case 'voxel'
    % Export beads=1, void=0 as NRRD header + gzip-compressed raw.
    dataPath = fullfile(p.paths.outDir,[p.out.baseName strrep('_voxelized','.','p') '.raw.gz']);
    hdrPath  = fullfile(p.paths.outDir,[p.out.baseName strrep('_voxelized','.','p') '.nhdr']);
    
    % % beads=1, void=0
    % pad = p.field.padVox;
    % Vout_xyz = uint8(F_beads >= isoLevel); % includes padding
    % if pad > 0
    %     Vout_xyz = Vout_xyz((pad+1):(end-pad), (pad+1):(end-pad), (pad+1):(end-pad));
    % end
    
    writeRawAndHeader(solid, voxSize, dataPath, hdrPath, 'nhdr_gz');
    
    fprintf('Export done:\n  %s\n  %s\n', hdrPath, dataPath);
    return
end

%% ================== STEP 5b: EXPORT RAW + META (FOR PY) ==================
switch p.mesh.backend
    case 'python'
        rawFile  = fullfile(p.paths.outDir, [p.out.baseName '_F_nxnyz.raw']);
        metaFile = fullfile(p.paths.outDir, [p.out.baseName '_meta.txt']);
        
        fid = fopen(rawFile,'w');  assert(fid>0,'Cannot write: %s', rawFile);
        fwrite(fid, permute(single(F), [2 1 3]), 'single');
        fclose(fid);
        
        fid = fopen(metaFile,'w'); assert(fid>0,'Cannot write: %s', metaFile);
        fprintf(fid, '%d %d %d\n', nx, ny, nz);
        fprintf(fid, '%.15g\n', voxSize);
        fprintf(fid, '%.15g %.15g %.15g\n', origin(1), origin(2), origin(3));
        fprintf(fid, '%.15g\n', isoLevel);
        fclose(fid);
        
        fprintf('Wrote:\n  %s\n  %s\n', rawFile, metaFile);
end

%% ================= STEP 5c: EXPORT SCALAR FIELD F FOR 3D SLICER =========
% % Export float32 implicit field F (isoLevel=0) as NRRD header + gzip raw.
% 
% pad = p.field.padVox;
% 
% Fout = single(F); % <-- implicit field you want to iso-surface
% if pad > 0
%     Fout = Fout((pad+1):(end-pad), (pad+1):(end-pad), (pad+1):(end-pad));
% end
% 
% dataPathF = fullfile(p.paths.outDir, [p.out.baseName '_F.raw.gz']);
% hdrPathF  = fullfile(p.paths.outDir, [p.out.baseName '_F.nhdr']);
% 
% originF = origin + pad*voxSize; % after trimming padding, this is typically [0 0 0]
% 
% writeRawAndHeaderFloat(Fout, voxSize, originF, dataPathF, hdrPathF);
% 
% fprintf('Exported scalar F for Slicer:\n  %s\n  %s\n', hdrPathF, dataPathF);

%% ================ STEP 6: MESH + WRITE STL and 3MF ======================
switch lower(p.mesh.backend)

    % ============================ MATLAB backend =========================
    case 'matlab'

        switch lower(p.mesh.method)

            % -------- extractIsosurface (Medical Imaging Toolbox) --------
            case 'extractisosurface'
                assert(exist('extractIsosurface','file')==2, ...
                    'extractIsosurface not found (requires Medical Imaging Toolbox).');
                t0 = tic;
                [FACES, V_ijk] = extractIsosurface(single(F), single(isoLevel)); % verts in intrinsic ijk
                fprintf('extractIsosurface done in %.2fs\n', toc(t0));

                VERT = [ ...
                    origin(1) + (double(V_ijk(:,1)) - 1) * voxSize, ...
                    origin(2) + (double(V_ijk(:,2)) - 1) * voxSize, ...
                    origin(3) + (double(V_ijk(:,3)) - 1) * voxSize  ];

            % --------------------- isosurface (built-in) ----------------
            case 'isosurface'
                t0 = tic;
                [FACES, V_ijk] = isosurface(single(F), single(isoLevel)); % verts in intrinsic ijk
                fprintf('isosurface done in %.2fs\n', toc(t0));

                %FACES = FACES(:,[1 3 2]);           % flip normals

                VERT = [ ...
                    origin(1) + (double(V_ijk(:,1)) - 1) * voxSize, ...
                    origin(2) + (double(V_ijk(:,2)) - 1) * voxSize, ...
                    origin(3) + (double(V_ijk(:,3)) - 1) * voxSize  ];

            % ----------------------- iso2mesh (MATLAB) -------------------
            case 'iso2mesh'
                assert(exist('v2s','file')==2, ...
                    'iso2mesh v2s not found on MATLAB path (need iso2mesh toolbox).');

                % v2s expects a scalar/level-set field; isoLevel=0 extracts the interface
                img = single(F);
                %img = permute(single(F), [2 1 3]);


                opt = struct();
                opt.angbound = p.mesh.iso2mesh.angbound_deg;   % deg
                opt.radbound = p.mesh.iso2mesh.radbound;       % vox units
                opt.distbound= p.mesh.iso2mesh.distbound;      % vox units
                opt.maxnode  = p.mesh.iso2mesh.maxnode;

                t0 = tic;
                [NODE, ELEM] = v2s(img, single(isoLevel), opt, 'cgalsurf'); % ELEM: Nx4 (last col region id)

                fprintf('iso2mesh v2s(cgalsurf) done in %.2fs\n', toc(t0));

                if size(ELEM,2) > 3, ELEM = ELEM(:,1:3); end
                FACES = double(ELEM(:,1:3));

                NODE = double(NODE);

                % Detect 0-based vs 1-based NODE coords
                if min(NODE,[],'all') >= 1 - 1e-6
                    NODE = NODE - 1;
                end

                % Map voxel coords -> physical mm coords
                origin_corner0 = origin - 0.5*voxSize;
                VERT = origin_corner0 + NODE * voxSize;

                fprintf('Mesh bbox [mm]: x[%.3g %.3g] y[%.3g %.3g] z[%.3g %.3g]\n', ...
                    min(VERT(:,1)), max(VERT(:,1)), ...
                    min(VERT(:,2)), max(VERT(:,2)), ...
                    min(VERT(:,3)), max(VERT(:,3)));

            otherwise
                error('Unknown MATLAB mesh method: %s', p.mesh.method);
        end

        % ------------------- optional post-processing --------------------
        if p.mesh.doReducePatch
            Sred = reducepatch(struct('faces',FACES,'vertices',VERT), p.mesh.reduceFactor);
            FACES = Sred.faces;
            VERT  = Sred.vertices;
        end

        if p.mesh.doRepair
            assert(exist('meshcheckrepair','file')==2, ...
                'p.mesh.doRepair requires iso2mesh meshcheckrepair on path.');
            [VERT, FACES] = meshcheckrepair(VERT, FACES, 'meshfix');
        end

        % -------------------------- Stats ----------------------------
        TR = triangulation(double(FACES), double(VERT));
        
        VE = TR.Points;                 % Nv x 3
        FC = TR.ConnectivityList;       % Nf x 3
        
        e1 = VE(FC(:,2),:) - VE(FC(:,1),:);
        e2 = VE(FC(:,3),:) - VE(FC(:,1),:);
        
        A = 0.5 * sqrt(sum(cross(e1,e2,2).^2, 2));   % Nf x 1 triangle areas
        
        fprintf('Degenerate faces (A<1e-12): %d\n', nnz(A < 1e-12));
        
        FB = freeBoundary(TR);  % for 3D triangulation: free boundary FACETS (triangles)
        fprintf('Free-boundary facets: %d\n', size(FB,1));
        
        E = sort([FC(:,[1 2]); FC(:,[2 3]); FC(:,[3 1])], 2);
        [~,~,ic] = unique(E,'rows');
        cnt = accumarray(ic,1);
        fprintf('Boundary edges: %d\n', nnz(cnt==1));
        fprintf('Non-manifold edges: %d\n', nnz(cnt>2));
        fprintf('Manifold+closed: %d\n', (nnz(cnt==1)==0) && (nnz(cnt>2)==0));
        
        % -------------------------- write STL ----------------------------
        if p.out.saveSTL
            fprintf('Writing STL: %s\n', p.out.stl);
            stlwrite(TR, p.out.stl,'binary');
            fprintf('DONE.\n');
        end
        
        % -------------------------- write 3MF ----------------------------
        if p.out.save3MF
            fprintf('Writing 3MF: %s\n', p.out.threemf);
            write3mf(p.out.threemf, TR.Points, TR.ConnectivityList);
            fprintf('DONE.\n');
        end

    % ============================ PYTHON backend =========================
    case 'python'
        assert(isfile(p.py.script), 'Python script not found: %s', p.py.script);
    
        %cmd = sprintf('"%s" "%s" "%s" "%s" "%s" --method %s', ...
        %    p.py.exe, p.py.script, rawFile, metaFile, p.out.stl, p.mesh.method);

        cmd = sprintf('"%s" "%s" "%s" "%s" "%s" --method %s --out_3mf "%s"', ...
            p.py.exe, p.py.script, rawFile, metaFile, p.out.stl, p.mesh.method, p.out.threemf);
    
        if strcmpi(p.mesh.method,'iso2mesh')
            cmd = sprintf('%s --angbound_deg %.15g --radbound %.15g --distbound %.15g --maxnode %d', ...
                cmd, p.mesh.iso2mesh.angbound_deg, p.mesh.iso2mesh.radbound, p.mesh.iso2mesh.distbound, round(p.mesh.iso2mesh.maxnode));
        end
    
        fprintf('Python cmd:\n%s\n', cmd);
        [status, out] = system(cmd);
        assert(status==0, 'Python meshing failed:\n%s', out);
        fprintf('%s\n', out);
        fprintf('Wrote STL: %s\n', p.out.stl);

    otherwise
        error('p.mesh.backend must be ''matlab'' or ''python''.');
end

%% =================== STEP 7: WRITE INFO (OPTIONAL) ======================
if p.out.writeInfoFile
    infoFile = fullfile(p.paths.outDir, [p.out.baseName '_info.txt']);
    fid = fopen(infoFile,'w'); assert(fid>0,'Cannot write: %s', infoFile);
    fprintf(fid,'baseName: %s\n', p.out.baseName);
    fprintf(fid,'packing: %s\n', p.paths.packing);
    fprintf(fid,'field.mode: %s\n', p.field.mode);
    fprintf(fid,'field.exportWhat: %s\n', p.field.exportWhat);
    fprintf(fid,'field.exportMode: %s\n', p.field.exportMode);
    fprintf(fid,'voxSize_snapped_mm: %.15g\n', p.grid.voxSize_mm_snapped);
    fprintf(fid,'nx ny nz: %d %d %d\n', nx, ny, nz);
    fprintf(fid,'origin_mm: %.15g %.15g %.15g\n', origin(1),origin(2),origin(3));
    fprintf(fid,'isoLevel: %.15g\n', isoLevel);
    fprintf(fid,'mesh.backend: %s\n', p.mesh.backend);
    fprintf(fid,'mesh.method: %s\n', p.mesh.method);
    fprintf(fid,'iso2mesh.angbound_deg: %.15g\n', p.mesh.iso2mesh.angbound_deg);
    fprintf(fid,'iso2mesh.radbound: %.15g\n', p.mesh.iso2mesh.radbound);
    fprintf(fid,'iso2mesh.distbound: %.15g\n', p.mesh.iso2mesh.distbound);
    fprintf(fid,'iso2mesh.maxnode: %.15g\n', p.mesh.iso2mesh.maxnode);
    fclose(fid);
    fprintf('Wrote info: %s\n', infoFile);
end


%% time out
toc
%% ========================= LOCAL FUNCTIONS ==============================
function writeRawAndHeader(Vout_xyz, voxSize, dataPath, hdrPath, exportFormat)
% Vout_xyz must be [nx ny nz] uint8

assert(isa(Vout_xyz,'uint8'),'Vout must be uint8');
[nxE, nyE, nzE] = size(Vout_xyz);

switch lower(exportFormat)
    case 'nhdr_gz'
        rawTmp = strrep(dataPath, '.raw.gz', '.raw');
        fid = fopen(rawTmp,'w'); assert(fid>0,'Cannot write %s',rawTmp);
        fwrite(fid, Vout_xyz, 'uint8');
        fclose(fid);

        gzip(rawTmp);
        delete(rawTmp);

        fid = fopen(hdrPath,'w'); assert(fid>0,'Cannot write %s',hdrPath);
        fprintf(fid,'NRRD0005\n');
        fprintf(fid,'type: uchar\n');
        fprintf(fid,'dimension: 3\n');
        fprintf(fid,'sizes: %d %d %d\n', nxE, nyE, nzE);
        fprintf(fid,'encoding: gzip\n');
        fprintf(fid,'endian: little\n');
        fprintf(fid,'space: right-anterior-superior\n');
        fprintf(fid,'space directions: (%.9g,0,0) (0,%.9g,0) (0,0,%.9g)\n', voxSize, voxSize, voxSize);
        fprintf(fid,'space origin: (0,0,0)\n');
        [~,df,ext] = fileparts(dataPath);
        fprintf(fid,'data file: %s%s\n', df, ext);
        fclose(fid);

    case 'mhd_raw'
        fid = fopen(dataPath,'w'); assert(fid>0,'Cannot write %s',dataPath);
        fwrite(fid, Vout_xyz, 'uint8');
        fclose(fid);

        fid = fopen(hdrPath,'w'); assert(fid>0,'Cannot write %s',hdrPath);
        fprintf(fid,'ObjectType = Image\n');
        fprintf(fid,'NDims = 3\n');
        fprintf(fid,'BinaryData = True\n');
        fprintf(fid,'BinaryDataByteOrderMSB = False\n');
        fprintf(fid,'CompressedData = False\n');
        fprintf(fid,'TransformMatrix = 1 0 0 0 1 0 0 0 1\n');
        fprintf(fid,'Offset = 0 0 0\n');
        fprintf(fid,'CenterOfRotation = 0 0 0\n');
        fprintf(fid,'AnatomicalOrientation = RAI\n');
        fprintf(fid,'ElementSpacing = %.9g %.9g %.9g\n', voxSize, voxSize, voxSize);
        fprintf(fid,'DimSize = %d %d %d\n', nxE, nyE, nzE);
        fprintf(fid,'ElementType = MET_UCHAR\n');
        [~,df,ext] = fileparts(dataPath);
        fprintf(fid,'ElementDataFile = %s%s\n', df, ext);
        fclose(fid);

    otherwise
        error('Unsupported exportFormat.');
end
end


function writeRawAndHeaderFloat(Fout_xyz, voxSize, origin_mm, dataPath, hdrPath)
% Fout_xyz must be [nx ny nz] single

assert(isa(Fout_xyz,'single'),'Fout must be single');
[nxE, nyE, nzE] = size(Fout_xyz);

rawTmp = strrep(dataPath, '.raw.gz', '.raw');
fid = fopen(rawTmp,'w'); assert(fid>0,'Cannot write %s',rawTmp);
fwrite(fid, Fout_xyz, 'single');
fclose(fid);

gzip(rawTmp);
delete(rawTmp);

fid = fopen(hdrPath,'w'); assert(fid>0,'Cannot write %s',hdrPath);
fprintf(fid,'NRRD0005\n');
fprintf(fid,'type: float\n');
fprintf(fid,'dimension: 3\n');
fprintf(fid,'sizes: %d %d %d\n', nxE, nyE, nzE);
fprintf(fid,'encoding: gzip\n');
fprintf(fid,'endian: little\n');
fprintf(fid,'space: right-anterior-superior\n');
fprintf(fid,'space directions: (%.9g,0,0) (0,%.9g,0) (0,0,%.9g)\n', voxSize, voxSize, voxSize);
fprintf(fid,'space origin: (%.9g,%.9g,%.9g)\n', origin_mm(1), origin_mm(2), origin_mm(3));
[~,df,ext] = fileparts(dataPath);
fprintf(fid,'data file: %s%s\n', df, ext);
fclose(fid);
end
