%% generate_reference.m — regenerate tests/fixtures/reference.3mf (fully scripted).
%
% Run this file in MATLAB (any cwd). It executes an *unmodified-on-disk*
% matlab/legacy/RCPS_v4.m with the canonical reference configuration applied
% programmatically — no manual editing of RCPS_v4.m required or wanted.
%
% The two load-bearing overrides vs. the RCPS_v4.m defaults:
%
%   p.field.exportMode = 'facility'  % 1x1x1 facility => periodic ghost
%   p.field.ghostTiles = 1           %   spheres ARE generated.
%
% WHY: in RCPS_v4.m, ghost (periodic-image) spheres are only created in
% 'facility' mode (line ~162); 'tile' mode meshes the raw spheres with no
% periodic images. Without ghosts, the tile loses the neighbour-sphere
% caps protruding through its faces: for data_example that is 5,592 mm^3,
% i.e. phi = 0.408 instead of the periodic bulk value 0.3633 (audited
% 2026-06-10 against the analytic packing). The 2026-06-05 reference was
% generated in 'tile' mode, which is why the Python comparison failed.
%
% KNOWN, ACCEPTED v4 BEHAVIOUR: the output mesh frame is rigidly shifted
% by exactly -1 voxel (-0.1 mm at vox=0.1) on every axis; the flush-cut
% faces land at [-0.1, 49.9] instead of the physical [0, 50]. Decision
% 2026-06-10: keep RCPS_v4.m as-is (it is the frozen legacy reference);
% the pytest suite detects and removes this rigid offset (see
% tests/test_pipeline_e2e.py::TestPythonMatchesMatlabReference). A
% cleaned MATLAB-only RCPS_v5 with the frame fixed at the source exists
% at matlab/RCPS_v5.m.
%
% Dependencies on the MATLAB path:
%   - iso2mesh toolbox (v2s/cgalsurf)
%   - write3mf  (https://github.com/cvergari/write3mf)
%   - Statistics and Machine Learning Toolbox (rangesearch)

% ---- locate repo (robust to cwd) ----
hereDir  = fileparts(mfilename('fullpath'));   % tests/fixtures
repoRoot = fullfile(hereDir, '..', '..');
legacyDir = fullfile(repoRoot, 'matlab', 'legacy');
dataDir  = fullfile(repoRoot, 'data_example');

src = fileread(fullfile(legacyDir, 'RCPS_v4.m'));

% ---- programmatic overrides (anchored to assignment lines only) ----
esc = @(pth) strrep(pth, '''', '''''');
ovr = {
  '^clear; clc; close all', ...
      '% (clear/clc stripped — set by generate_reference.m)';
  '^p\.paths\.root\s*=[^\n]*', ...
      sprintf('p.paths.root = ''%s%s'';  %% set by generate_reference.m', esc(dataDir), filesep);
  '^p\.paths\.outDir\s*=[^\n]*', ...
      sprintf('p.paths.outDir = ''%s%s'';  %% set by generate_reference.m', esc(hereDir), filesep);
  '^p\.out\.baseName\s*=\s*''[^\n]*', ...
      'p.out.baseName = ''reference'';  % set by generate_reference.m';
  '^p\.field\.exportMode\s*=[^\n]*', ...
      'p.field.exportMode = ''facility'';  % set by generate_reference.m (1x1x1 + periodic ghosts)';
  '^p\.field\.ghostTiles\s*=[^\n]*', ...
      'p.field.ghostTiles = 1;  % set by generate_reference.m';
  '^p\.field\.exportWhat\s*=[^\n]*', ...
      'p.field.exportWhat = ''beads'';  % set by generate_reference.m';
  '^p\.field\.keepSides\s*=[^\n]*', ...
      'p.field.keepSides = {};  % set by generate_reference.m (ALL faces cut flush)';
  '^p\.grid\.voxSize_mm\s*=[^\n]*', ...
      'p.grid.voxSize_mm = 0.1;  % set by generate_reference.m';
  '^p\.spheres\.expansion_factor\s*=[^\n]*', ...
      'p.spheres.expansion_factor = 1.00;  % set by generate_reference.m';
  '^p\.bridge\.mode\s*=[^\n]*', ...
      'p.bridge.mode = ''cylinders'';  % set by generate_reference.m';
  '^p\.bridge\.radiusFrac\s*=[^\n]*', ...
      'p.bridge.radiusFrac = 0.15;  % set by generate_reference.m';
};
% Every canonical parameter is pinned here so the reference does not
% depend on the current editing state of the legacy script (lesson from
% 2026-06-10: the script previously relied on v4's on-disk defaults for
% keepSides/vox/bridges, which differ between v4 snapshots).
for k = 1:size(ovr, 1)
    src = regexprep(src, ovr{k,1}, ovr{k,2}, 'lineanchors', 'once');
end
assert(numel(strfind(src, 'set by generate_reference.m')) == size(ovr, 1) - 1, ...
    'generate_reference: an override pattern did not match RCPS_v4.m (expected %d "set by" markers) — check the regexes above.', size(ovr, 1) - 1);

tmpScript = fullfile(hereDir, 'reference_run_tmp.m');
fid = fopen(tmpScript, 'w'); fwrite(fid, src); fclose(fid);

fprintf('=== generate_reference: running RCPS_v4 (facility 1x1x1, ghosts ON) ===\n');
run(tmpScript);   % leaves the `p` struct in this workspace

% ---- append an audit block to reference_info.txt (v4 omits these) ----
hereDir = fileparts(mfilename('fullpath'));    % re-derive, belt and braces
delete(fullfile(hereDir, 'reference_run_tmp.m'));

fid = fopen(fullfile(hereDir, 'reference_info.txt'), 'a');
fprintf(fid, 'field.ghostTiles: %d\n',              p.field.ghostTiles);
fprintf(fid, 'spheres.expansion_factor: %.15g\n',   p.spheres.expansion_factor);
fprintf(fid, 'spheres.contactTol_mm: %.15g\n',      p.spheres.contactTol_mm);
fprintf(fid, 'bridge.mode: %s\n',                   p.bridge.mode);
fprintf(fid, 'bridge.radiusFrac: %.15g\n',          p.bridge.radiusFrac);
fprintf(fid, 'generated: %s | MATLAB %s | via generate_reference.m\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM'), version);
fclose(fid);

% ---- expectations ----
fprintf('\n=== generate_reference: done ===\n');
fprintf('Check the "Mesh bbox" line printed above: it should read\n');
fprintf('  x[-0.1 49.9] y[-0.1 49.9] z[-0.1 49.9]   (the known v4 frame offset).\n');
fprintf('Expected pytest comparison values after this regeneration:\n');
fprintf('  solid volume  ~ 79.6e3 mm^3  (Python: 79,593; analytic periodic: 79,583)\n');
fprintf('  porosity      ~ 0.3633       (the no-ghost tile-mode reference gave 0.4073)\n');
fprintf('Now run:\n');
fprintf('  pytest "tests/test_pipeline_e2e.py::TestPythonMatchesMatlabReference" -v\n');
fprintf('Tip: export RCPS_E2E_CACHE_DIR=~/.cache/rcps_e2e to reuse the 80-min Python run.\n');
