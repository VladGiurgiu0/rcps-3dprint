%% =========================== RCPS_v5.m =====================================
% MATLAB-only successor of RCPS_v4.m (2026-06-10). Pipeline:
%   load packing.xyzd -> periodic replication -> ICSG signed field
%   -> optional cylinder bridges -> iso2mesh (v2s/cgalsurf) -> .3mf
%
% Changes vs RCPS_v4.m
% --------------------
% 1. FRAME FIX: v4's NODE->mm mapping was rigidly off by -1 voxel (flush-cut
%    faces landed at [-vox, L-vox] instead of [0, L]). v5 corrects the
%    mapping and ASSERTS the frame after meshing: with no keepSides, the
%    mesh bbox must equal [0,L]x[0,H]x[0,W] to within 1e-3 mm.
% 2. GHOSTS ALWAYS: v4 only generated periodic ghost spheres in 'facility'
%    mode; 'tile' mode silently dropped them, biasing boundary porosity
%    (phi 0.408 instead of bulk 0.3633 on data_example). v5 replaces
%    exportMode with p.geom.nTiles = [nx ny nz] and ALWAYS replicates
%    periodic images. [1 1 1] is a single tile; [4 4 1] is a facility
%    printed as one piece.
% 3. MATLAB-ONLY: the python-calling backend is removed. Python users
%    should use the `rcps` package (pip install -e ., `rcps-build`),
%    which is the maintained reference implementation.
% 4. Dropped legacy paths: 'voxel' field mode, extractIsosurface /
%    isosurface methods, NRRD/Slicer exports, reducepatch, meshcheckrepair.
%
% keepSides and interlocking tiles
% --------------------------------
% A face listed in p.field.keepSides (RAS: R/L/A/P/S/I) is NOT cut and
% spheres whose center lies beyond it are dropped. Two adjacent tiles that
% both "keep" their shared face therefore interlock: each contains exactly
% the spheres whose centers it owns, and the packing's periodicity makes
% the protrusions of one fit the cavities of the other.
%
% Dependencies: iso2mesh toolbox (v2s), write3mf, Statistics & ML Toolbox
% (rangesearch; only when bridges are enabled).
% ===========================================================================

clear; clc; close all

%% =========================== USER INPUTS ================================
p = struct();

% ------------------------------ Paths -----------------------------------
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end          % run-by-cell fallback
repoRoot = fullfile(thisDir, '..');

p.paths.packing = fullfile(repoRoot, 'data_example', 'packing.xyzd');
p.paths.outDir  = fullfile(repoRoot, 'output');

% ------------------------------ Geometry --------------------------------
p.geom.L_tile = 50;  p.geom.H_tile = 50;  p.geom.W_tile = 50;  % [mm] period of the packing
p.geom.nTiles = [1 1 1];   % tiles per axis; domain = nTiles .* tile size

% --------------------------- Field generation ---------------------------
p.field.exportWhat = 'beads';  % 'beads' only ('pore' is NOT validated in MATLAB -> use Python rcps)
p.field.ghostTiles = 1;        % layers of periodic images beyond the domain (1 - recommended)
p.field.padVox     = 1;        % grid padding so box walls close cleanly (1 - recommended)
p.field.bandVox    = 3;        % ICSG narrow band half-width in voxels (3 - recommended)
p.field.keepSides  = {};       % e.g. {'R','S'}: RAS faces where beads are NOT cut

% ------------------------------ Spheres ---------------------------------
p.spheres.expansion_factor = 1.00;   % print diameter multiplier (1 - recommended)
p.spheres.contactTol_mm    = 0.20;   % contact detection gap tolerance (bridges)

% ------------------------------ Bridges ---------------------------------
p.bridge.mode       = 'cylinders';   % 'none' | 'cylinders'
p.bridge.radiusFrac = 0.15;          % cyl radius = radiusFrac * min(r_i, r_j)

% -------------------------------- Grid ----------------------------------
p.grid.voxSize_mm = 0.1;             % target voxel size [mm] (snapped to divide L)

% ------------------------------- Meshing --------------------------------
p.mesh.iso2mesh.angbound_deg = 25;    % min triangle angle [deg]
p.mesh.iso2mesh.radbound     = 1.0;   % max Delaunay circumradius [vox]
p.mesh.iso2mesh.distbound    = 0.10;  % max surface deviation [vox]
p.mesh.iso2mesh.maxnode      = 2e8;   % hard cap on surface nodes

% ------------------------------- Output ---------------------------------
p.out.baseName      = 'rcps_v5_tile';
p.out.save3MF       = true;
p.out.saveSTL       = false;
p.out.writeInfoFile = true;

%% ========================= STEP 1: LOAD PACKING =========================
tic
fprintf('Loading packing: %s\n', p.paths.packing);
fid = fopen(p.paths.packing, 'r', 'ieee-le');
assert(fid > 0, 'Cannot open file: %s', p.paths.packing);
raw = fread(fid, Inf, 'double=>double');
fclose(fid);
assert(mod(numel(raw), 4) == 0, 'Invalid xyzd stream (count not divisible by 4).');

S  = reshape(raw, 4, []).';   % [x y z d]
x0 = S(:,1); y0 = S(:,2); z0 = S(:,3); d0 = S(:,4);
N0 = size(S, 1);
fprintf('Loaded spheres: N = %d, d = %.4f..%.4f mm\n', N0, min(d0), max(d0));

%% ============ STEP 2: DOMAIN + PERIODIC REPLICATION (ALWAYS) ============
nT = p.geom.nTiles;
assert(numel(nT) == 3 && all(nT >= 1) && all(nT == round(nT)), ...
    'p.geom.nTiles must be three positive integers.');

dom.L = nT(1) * p.geom.L_tile;
dom.H = nT(2) * p.geom.H_tile;
dom.W = nT(3) * p.geom.W_tile;

g = p.field.ghostTiles;
txList = (-g):(nT(1)-1+g);
tyList = (-g):(nT(2)-1+g);
tzList = (-g):(nT(3)-1+g);
nCopies = numel(txList) * numel(tyList) * numel(tzList);

x = repmat(x0, nCopies, 1);
y = repmat(y0, nCopies, 1);
z = repmat(z0, nCopies, 1);
d = repmat(d0, nCopies, 1);

cnt = 0;
for tz = tzList
    for ty = tyList
        for tx = txList
            idx = (1:N0) + cnt*N0;
            x(idx) = x(idx) + tx * p.geom.L_tile;
            y(idx) = y(idx) + ty * p.geom.H_tile;
            z(idx) = z(idx) + tz * p.geom.W_tile;
            cnt = cnt + 1;
        end
    end
end

% expansion (nominal -> print)
r_nom = 0.5 * d;
r     = r_nom * p.spheres.expansion_factor;
d     = 2 * r;

Ns = numel(r);
fprintf('Domain: L=%.4g H=%.4g W=%.4g mm | tiles: %d %d %d | spheres incl. ghosts: %d\n', ...
    dom.L, dom.H, dom.W, nT(1), nT(2), nT(3), Ns);

%% ================== STEP 3: SNAP GRID (DOMAIN-DIVISIBLE) ================
voxSize = p.grid.voxSize_mm;

nx0 = max(8, round(dom.L / voxSize));
voxSize = dom.L / nx0;                       % snap so L is divisible

ny0 = round(dom.H / voxSize);
nz0 = round(dom.W / voxSize);
assert(abs(dom.H/voxSize - ny0) < 1e-10, 'H not divisible by voxSize after snap');
assert(abs(dom.W/voxSize - nz0) < 1e-10, 'W not divisible by voxSize after snap');

p.grid.voxSize_mm_snapped = voxSize;
fprintf('Grid (snapped): nx0=%d ny0=%d nz0=%d | voxSize=%.6g mm\n', nx0, ny0, nz0, voxSize);

%% ===================== STEP 4: BUILD SCALAR FIELD =======================
padVox     = p.field.padVox;
bandVoxEff = max(2, p.field.bandVox);
bandDist   = bandVoxEff * voxSize;

haveKeep = ~isempty(p.field.keepSides);
if haveKeep
    rMax = max(r);
    padVox = padVox + ceil((rMax + bandDist)/voxSize) + 1;
end

nx = nx0 + 2*padVox;
ny = ny0 + 2*padVox;
nz = nz0 + 2*padVox;

nVox = double(nx) * double(ny) * double(nz);
if nVox > 5e8
    error(['Dense grid too large: nx*ny*nz = %.3g voxels (%.2f GiB for one float32 field). ', ...
           'Reduce resolution or domain size.'], nVox, 4*nVox/1024^3);
end

origin = [-padVox, -padVox, -padVox] * voxSize;   % coordinate of voxel (1,1,1) center

xVec = origin(1) + (0:nx-1)*voxSize;
yVec = origin(2) + (0:ny-1)*voxSize;
zVec = origin(3) + (0:nz-1)*voxSize;

% ---- box SDF (negative inside [0,L]x[0,H]x[0,W]); kept faces pushed out ----
xmin = 0; xmax = dom.L;
ymin = 0; ymax = dom.H;
zmin = 0; zmax = dom.W;

if haveKeep
    extMm = 2*(max(r) + bandDist) + 2*voxSize;
    ks = upper(string(p.field.keepSides));
    if any(ks=="L"), xmin = xmin - extMm; end
    if any(ks=="R"), xmax = xmax + extMm; end
    if any(ks=="P"), ymin = ymin - extMm; end
    if any(ks=="A"), ymax = ymax + extMm; end
    if any(ks=="I"), zmin = zmin - extMm; end
    if any(ks=="S"), zmax = zmax + extMm; end
end

cbx = (xmin + xmax)/2;  bx = (xmax - xmin)/2;
cby = (ymin + ymax)/2;  by = (ymax - ymin)/2;
cbz = (zmin + zmax)/2;  bz = (zmax - zmin)/2;
qx = abs(xVec - cbx) - bx;
qy = abs(yVec - cby) - by;
qz = abs(zVec - cbz) - bz;

QX = reshape(single(qx), [], 1, 1);
QY = reshape(single(qy), 1, [], 1);
QZ = reshape(single(qz), 1, 1, []);

QXp = max(QX, 0); QYp = max(QY, 0); QZp = max(QZ, 0);
outside = sqrt(QXp.^2 + QYp.^2 + QZp.^2);
inside  = min(max(max(QX, QY), QZ), 0);
F_box   = outside + inside;      % strict SDF, negative inside

% ---- cull spheres that cannot influence the padded grid ----
radCull = r + bandDist;
keep = (x + radCull >= xVec(1)) & (x - radCull <= xVec(end)) & ...
       (y + radCull >= yVec(1)) & (y - radCull <= yVec(end)) & ...
       (z + radCull >= zVec(1)) & (z - radCull <= zVec(end));

NsBefore = Ns;
x = x(keep); y = y(keep); z = z(keep); r = r(keep); d = d(keep);
Ns = numel(r);
fprintf('Culled spheres: %d -> %d kept\n', NsBefore, Ns);

% ---- ownership: a kept face drops spheres whose center lies beyond it ----
% (this is what makes adjacent keep-face tiles interlock)
if haveKeep
    tol = 1e-12;
    keep2 = true(size(x));
    if any(ks=="L"), keep2 = keep2 & (x >= 0     - tol); end
    if any(ks=="R"), keep2 = keep2 & (x <= dom.L + tol); end
    if any(ks=="P"), keep2 = keep2 & (y >= 0     - tol); end
    if any(ks=="A"), keep2 = keep2 & (y <= dom.H + tol); end
    if any(ks=="I"), keep2 = keep2 & (z >= 0     - tol); end
    if any(ks=="S"), keep2 = keep2 & (z <= dom.W + tol); end

    NsBefore = Ns;
    x = x(keep2); y = y(keep2); z = z(keep2); r = r(keep2); d = d(keep2);
    Ns = numel(r);
    fprintf('keepSides ownership filter: %d -> %d kept\n', NsBefore, Ns);
end

% ---- ICSG narrow-band union of spheres (negative inside) ----
F_beads = single(bandDist) * ones(nx, ny, nz, 'single');

t0 = tic;
for i = 1:Ns
    xc = x(i); yc = y(i); zc = z(i); R = r(i);
    rad = R + bandDist;

    ix1 = max(floor((xc - rad - origin(1))/voxSize) + 1, 1);
    ix2 = min(ceil( (xc + rad - origin(1))/voxSize) + 1, nx);
    iy1 = max(floor((yc - rad - origin(2))/voxSize) + 1, 1);
    iy2 = min(ceil( (yc + rad - origin(2))/voxSize) + 1, ny);
    iz1 = max(floor((zc - rad - origin(3))/voxSize) + 1, 1);
    iz2 = min(ceil( (zc + rad - origin(3))/voxSize) + 1, nz);
    if (ix1 > ix2) || (iy1 > iy2) || (iz1 > iz2), continue; end

    [XX, YY, ZZ] = ndgrid(single(xVec(ix1:ix2) - xc), ...
                          single(yVec(iy1:iy2) - yc), ...
                          single(zVec(iz1:iz2) - zc));
    local = sqrt(XX.^2 + YY.^2 + ZZ.^2) - single(R);

    blk = F_beads(ix1:ix2, iy1:iy2, iz1:iz2);
    F_beads(ix1:ix2, iy1:iy2, iz1:iz2) = min(blk, local);
end
fprintf('ICSG spheres done in %.2fs\n', toc(t0));

%% ===================== STEP 4b: OPTIONAL BRIDGES ========================
if ~strcmpi(p.bridge.mode, 'none')
    assert(exist('rangesearch', 'file') == 2, ...
        'Bridge mode requires rangesearch (Statistics and Machine Learning Toolbox).');

    ctr = [x y z];
    searchR = 2*max(r) + p.spheres.contactTol_mm;
    nbrs = rangesearch(ctr, ctr, searchR);

    nBridges = 0;
    t0 = tic;
    for i = 1:Ns
        ni = nbrs{i};
        ni = ni(ni > i);                       % upper triangle
        c1 = ctr(i,:);
        for kk = 1:numel(ni)
            j  = ni(kk);
            c2 = ctr(j,:);
            v  = c2 - c1;
            Lc = norm(v);
            if Lc <= 0, continue; end
            if Lc > (r(i) + r(j) + p.spheres.contactTol_mm), continue; end

            rcyl = p.bridge.radiusFrac * min(r(i), r(j));
            pad  = rcyl + bandDist;

            ix1 = max(floor((min(c1(1),c2(1)) - pad - origin(1))/voxSize) + 1, 1);
            ix2 = min(ceil( (max(c1(1),c2(1)) + pad - origin(1))/voxSize) + 1, nx);
            iy1 = max(floor((min(c1(2),c2(2)) - pad - origin(2))/voxSize) + 1, 1);
            iy2 = min(ceil( (max(c1(2),c2(2)) + pad - origin(2))/voxSize) + 1, ny);
            iz1 = max(floor((min(c1(3),c2(3)) - pad - origin(3))/voxSize) + 1, 1);
            iz2 = min(ceil( (max(c1(3),c2(3)) + pad - origin(3))/voxSize) + 1, nz);
            if (ix1 > ix2) || (iy1 > iy2) || (iz1 > iz2), continue; end

            [XX, YY, ZZ] = ndgrid(xVec(ix1:ix2), yVec(iy1:iy2), zVec(iz1:iz2));

            u  = v / Lc;
            dx = XX - c1(1); dy = YY - c1(2); dz = ZZ - c1(3);
            t  = dx*u(1) + dy*u(2) + dz*u(3);
            t  = max(0, min(Lc, t));
            ax = c1(1) + t*u(1);
            ay = c1(2) + t*u(2);
            az = c1(3) + t*u(3);
            dist = sqrt((XX-ax).^2 + (YY-ay).^2 + (ZZ-az).^2);

            % NOTE: the -0.25*voxSize dilation is inherited from RCPS_v4
            % (kept for parity with the validated reference behaviour).
            local = single(dist) - single(rcyl) - 0.25*voxSize;
            blk   = F_beads(ix1:ix2, iy1:iy2, iz1:iz2);
            F_beads(ix1:ix2, iy1:iy2, iz1:iz2) = min(blk, local);
            nBridges = nBridges + 1;
        end
    end
    fprintf('Bridges: %d cylinders in %.2fs\n', nBridges, toc(t0));
end

%% ===================== STEP 4c: FINAL FIELD F ===========================
isoLevel = -1e-6 * voxSize;

F_beads = max(F_beads, F_box);     % confine to (possibly keep-extended) box

switch lower(p.field.exportWhat)
    case 'beads'
        F = F_beads;
    case 'pore'
        % Decision 2026-06-11: pore export is NOT validated in MATLAB
        % (v4's pore formula was sign-suspect; this path has never been
        % checked against a reference). Use the Python rcps package for
        % pore-space exports.
        error(['RCPS_v5: ''pore'' export is not validated in MATLAB. ', ...
               'Use the Python rcps package (rcps-build) for pore exports.']);
    otherwise
        error('p.field.exportWhat must be ''beads'' or ''pore''.');
end

fprintf('Field ready: size = [%d %d %d], isoLevel = %.3g\n', nx, ny, nz, isoLevel);

%% ===================== STEP 5: MESH (iso2mesh v2s) ======================
assert(exist('v2s', 'file') == 2, ...
    'iso2mesh v2s not found on MATLAB path (need iso2mesh toolbox).');

opt = struct();
opt.angbound  = p.mesh.iso2mesh.angbound_deg;
opt.radbound  = p.mesh.iso2mesh.radbound;
opt.distbound = p.mesh.iso2mesh.distbound;
opt.maxnode   = p.mesh.iso2mesh.maxnode;

t0 = tic;
[NODE, ELEM] = v2s(single(F), single(isoLevel), opt, 'cgalsurf');
fprintf('iso2mesh v2s(cgalsurf) done in %.2fs\n', toc(t0));

if size(ELEM, 2) > 3, ELEM = ELEM(:, 1:3); end
FACES = double(ELEM);
NODE  = double(NODE);

% Detect 0-based vs 1-based NODE coords (robust across iso2mesh versions).
if min(NODE, [], 'all') >= 1 - 1e-6
    NODE = NODE - 1;
end

% ---- NODE -> mm mapping (FRAME FIX vs v4) -------------------------------
% v4 used  origin - 0.5*voxSize + NODE*voxSize  which lands the flush-cut
% faces at [-vox, L-vox] (verified against the analytic packing,
% 2026-06-10). The +0.5 form below lands them at [0, L]; the assertion
% after meshing enforces this, so any future iso2mesh convention change
% fails loudly instead of silently shifting the part.
origin_corner = origin + 0.5*voxSize;
VERT = origin_corner + NODE * voxSize;

fprintf('Mesh bbox [mm]: x[%.6g %.6g] y[%.6g %.6g] z[%.6g %.6g]\n', ...
    min(VERT(:,1)), max(VERT(:,1)), min(VERT(:,2)), max(VERT(:,2)), ...
    min(VERT(:,3)), max(VERT(:,3)));

% ---- frame assertion (only valid when every face is cut flush) ----------
if ~haveKeep
    bbLo = min(VERT, [], 1);
    bbHi = max(VERT, [], 1);
    tolF = 1e-3;  % mm
    offLo = max(abs(bbLo - [0 0 0]));
    offHi = max(abs(bbHi - [dom.L dom.H dom.W]));
    assert(offLo < tolF && offHi < tolF, ...
        ['Frame assertion failed: mesh bbox [%.6g %.6g %.6g]..[%.6g %.6g %.6g] ', ...
         'deviates from [0 0 0]..[%.6g %.6g %.6g] by up to %.4g mm. ', ...
         'The NODE->mm mapping convention has drifted - do not print this part.'], ...
        bbLo(1), bbLo(2), bbLo(3), bbHi(1), bbHi(2), bbHi(3), ...
        dom.L, dom.H, dom.W, max(offLo, offHi));
    fprintf('Frame assertion PASSED: bbox = [0 %.4g] x [0 %.4g] x [0 %.4g] within %.3g mm\n', ...
        dom.L, dom.H, dom.W, tolF);
end

%% ========================= STEP 6: MESH STATS ===========================
TR = triangulation(FACES, VERT);
VE = TR.Points;
FC = TR.ConnectivityList;

e1 = VE(FC(:,2),:) - VE(FC(:,1),:);
e2 = VE(FC(:,3),:) - VE(FC(:,1),:);
A  = 0.5 * sqrt(sum(cross(e1, e2, 2).^2, 2));
fprintf('Vertices: %d | Faces: %d | Degenerate faces (A<1e-12): %d\n', ...
    size(VE,1), size(FC,1), nnz(A < 1e-12));

E = sort([FC(:,[1 2]); FC(:,[2 3]); FC(:,[3 1])], 2);
[~, ~, ic] = unique(E, 'rows');
ecnt = accumarray(ic, 1);
fprintf('Boundary edges: %d | Non-manifold edges: %d | Manifold+closed: %d\n', ...
    nnz(ecnt==1), nnz(ecnt>2), (nnz(ecnt==1)==0) && (nnz(ecnt>2)==0));

% signed volume (divergence theorem) - compare with analytic expectation
Vsol = abs(sum(dot(VE(FC(:,1),:), cross(VE(FC(:,2),:), VE(FC(:,3),:), 2), 2)) / 6);
fprintf('Mesh solid volume: %.1f mm^3 | domain: %.1f mm^3 | solid fraction: %.4f | porosity: %.4f\n', ...
    Vsol, dom.L*dom.H*dom.W, Vsol/(dom.L*dom.H*dom.W), 1 - Vsol/(dom.L*dom.H*dom.W));

%% ========================= STEP 7: WRITE OUTPUT =========================
if ~exist(p.paths.outDir, 'dir'), mkdir(p.paths.outDir); end

if p.out.saveSTL
    stlPath = fullfile(p.paths.outDir, [p.out.baseName '.stl']);
    fprintf('Writing STL: %s\n', stlPath);
    stlwrite(TR, stlPath, 'binary');
end

if p.out.save3MF
    assert(exist('write3mf', 'file') == 2, ...
        'write3mf not found on path (https://github.com/cvergari/write3mf).');
    mfPath = fullfile(p.paths.outDir, [p.out.baseName '.3mf']);
    fprintf('Writing 3MF: %s\n', mfPath);
    write3mf(mfPath, TR.Points, TR.ConnectivityList);
end

if p.out.writeInfoFile
    infoFile = fullfile(p.paths.outDir, [p.out.baseName '_info.txt']);
    fid = fopen(infoFile, 'w'); assert(fid > 0, 'Cannot write: %s', infoFile);
    fprintf(fid, 'generator: RCPS_v5.m (frame-fixed; ghosts always on)\n');
    fprintf(fid, 'baseName: %s\n', p.out.baseName);
    fprintf(fid, 'packing: %s\n', p.paths.packing);
    fprintf(fid, 'geom.nTiles: %d %d %d\n', nT(1), nT(2), nT(3));
    fprintf(fid, 'domain_mm: %.15g %.15g %.15g\n', dom.L, dom.H, dom.W);
    fprintf(fid, 'field.exportWhat: %s\n', p.field.exportWhat);
    fprintf(fid, 'field.ghostTiles: %d\n', p.field.ghostTiles);
    fprintf(fid, 'field.keepSides: %s\n', strjoin(cellstr(string(p.field.keepSides)), ','));
    fprintf(fid, 'spheres.expansion_factor: %.15g\n', p.spheres.expansion_factor);
    fprintf(fid, 'spheres.contactTol_mm: %.15g\n', p.spheres.contactTol_mm);
    fprintf(fid, 'bridge.mode: %s\n', p.bridge.mode);
    fprintf(fid, 'bridge.radiusFrac: %.15g\n', p.bridge.radiusFrac);
    fprintf(fid, 'voxSize_snapped_mm: %.15g\n', voxSize);
    fprintf(fid, 'nx ny nz: %d %d %d\n', nx, ny, nz);
    fprintf(fid, 'origin_mm: %.15g %.15g %.15g\n', origin(1), origin(2), origin(3));
    fprintf(fid, 'isoLevel: %.15g\n', isoLevel);
    fprintf(fid, 'iso2mesh.angbound_deg: %.15g\n', p.mesh.iso2mesh.angbound_deg);
    fprintf(fid, 'iso2mesh.radbound: %.15g\n', p.mesh.iso2mesh.radbound);
    fprintf(fid, 'iso2mesh.distbound: %.15g\n', p.mesh.iso2mesh.distbound);
    fprintf(fid, 'iso2mesh.maxnode: %.15g\n', p.mesh.iso2mesh.maxnode);
    fprintf(fid, 'mesh_vertices: %d\n', size(VE,1));
    fprintf(fid, 'mesh_faces: %d\n', size(FC,1));
    fprintf(fid, 'mesh_solid_volume_mm3: %.6f\n', Vsol);
    fclose(fid);
    fprintf('Wrote info: %s\n', infoFile);
end

toc
