function chi03b_medi_runner(case_id)
% CHI03b SNU-LIST chi-separation MEDI runner.
%
% Container contract
% ------------------
% /project   APF project root, writable
% /toolbox   Chisep Toolbox v1.2.2, read-only
% /medi      MEDI toolbox, read-only
%
% Scientific policy
% -----------------
% - CHI02 R2-prime remains raw upstream.
% - At the solver boundary only:
%       non-finite R2-prime -> 0
%       negative R2-prime   -> 0
% - N_std is calculated with SNU-LIST Preprocessing4Phase.
% - CSF mask is generated with MEDI extract_CSF from R2STAR02.
% - Conventional chi-separation MEDI defaults:
%       Dr         = 137 Hz/ppm
%       lambda     = 1
%       lambda_CSF = 1
% - GRE06c STI-iLSQR QSM is supplied as option_data.qsm.
%
% Outputs are written in ppm.

addpath(genpath('/toolbox'));
addpath(genpath('/medi'));

root = '/project';
case_id = char(case_id);

chi02 = fullfile( ...
    root, 'derivatives', case_id, ...
    'chi_separation', 'chi02_prepare', 'qsm_ci_input');

phase_path = fullfile( ...
    root, 'derivatives', case_id, 'GRE', ...
    'phase_preparation_gre02', ...
    ['gre02_' case_id '_phase_rad_4d.nii.gz']);

r2star_path = fullfile( ...
    root, 'derivatives', case_id, ...
    'relaxometry', 'r2star', 'r2star02_loglinear', ...
    ['r2star02_' case_id '_r2star_s-1.nii.gz']);

qsm_path = fullfile( ...
    root, 'derivatives', case_id, 'GRE', ...
    '06c_sti_ilsqr', ...
    [case_id '_GRE06c_sti_ilsqr_chi_ppm.nii.gz']);

outdir = fullfile( ...
    root, 'derivatives', case_id, ...
    'chi_separation', 'chi03b_medi');

if ~isfolder(outdir)
    mkdir(outdir);
end

fprintf('=== CHI03b MEDI: %s ===\n', case_id);

%% Load CHI02 package

p = jsondecode(fileread(fullfile(chi02, 'params.json')));

localfield = double(niftiread(fullfile(chi02, 'localfield.nii.gz')));
r2prime    = double(niftiread(fullfile(chi02, 'r2prime.nii.gz')));
mag4       = double(niftiread(fullfile(chi02, 'magnitude.nii.gz')));
mask       = niftiread(fullfile(chi02, 'mask.nii.gz')) > 0.5;

phase4 = double(niftiread(phase_path));
r2star = double(niftiread(r2star_path));
qsm    = double(niftiread(qsm_path));

%% Basic geometry validation

target_size = size(mask);

assert(isequal(size(localfield), target_size), ...
    'Local-field geometry mismatch');
assert(isequal(size(r2prime), target_size), ...
    'R2-prime geometry mismatch');
assert(isequal(size(r2star), target_size), ...
    'R2-star geometry mismatch');
assert(isequal(size(qsm), target_size), ...
    'QSM geometry mismatch');
assert(isequal(size(mag4,1:3), target_size), ...
    'Magnitude geometry mismatch');
assert(isequal(size(phase4,1:3), target_size), ...
    'Phase geometry mismatch');
assert(size(mag4,4) == numel(p.TE), ...
    'Magnitude echo count does not match TE vector');
assert(size(phase4,4) == numel(p.TE), ...
    'Phase echo count does not match TE vector');

%% Solver inputs

mag = sqrt(sum(mag4.^2,4)) .* mask;

% Match CHI02/QSM-CI interface constant exactly.
CF = 42.5774e6 * p.B0;

local_field_hz = localfield * 1e-6 * CF;
local_field_hz = local_field_hz .* mask;

% Important:
% multiplication by zero does NOT sanitize NaN in MATLAB.
% Explicitly replace non-finite values before masking.
r2prime_solver = r2prime;

n_nonfinite_r2p = nnz(~isfinite(r2prime_solver));
n_nonfinite_r2p_mask = nnz(~isfinite(r2prime_solver(mask)));
n_negative_r2p_mask = nnz(r2prime_solver(mask) < 0);

r2prime_solver(~isfinite(r2prime_solver)) = 0;
r2prime_solver(r2prime_solver < 0) = 0;
r2prime_solver = r2prime_solver .* mask;

assert(all(isfinite(r2prime_solver(:))), ...
    'Sanitized R2-prime still contains non-finite values');

assert(all(isfinite(local_field_hz(mask))), ...
    'Local field contains non-finite values in solver mask');

assert(all(isfinite(mag(mask))) && all(mag(mask) > 0), ...
    'Magnitude invalid inside solver mask');

%% Official SNU-LIST noise weighting

fprintf('Calculating Preprocessing4Phase noise weighting...\n');
[~, N_std] = Preprocessing4Phase(mag4, phase4);

assert(all(isfinite(N_std(:))), ...
    'Preprocessing4Phase returned non-finite N_std');

%% MEDI CSF mask

voxel_size = double(p.voxel_size(:)');

% Default MEDI 10-mm SMV erosion eliminated the CSF reference in 8/10
% postmortem cases. Disable optional erosion while retaining the standard
% R2* < 5 s^-1 criterion.
csf_flag_erode = 0;
csf_thresh_R2s = 5;

mask_CSF = extract_CSF( ...
    r2star, ...
    mask, ...
    voxel_size, ...
    csf_flag_erode, ...
    csf_thresh_R2s);

mask_CSF = (mask_CSF > 0) & mask;

n_csf = nnz(mask_CSF);

if n_csf == 0
    error('CHI03b:EmptyCSFMask', ...
        'extract_CSF produced an empty CSF mask');
end

%% Parameters

params = struct();
params.b0_dir = double(p.B0_dir(:)');
params.CF = CF;
params.voxel_size = voxel_size;
params.TE = double(p.TE(:)');
params.lambda = 1;
params.lambda_CSF = 1;
params.Dr = 137;

option_data = struct();
option_data.qsm = qsm .* mask;
option_data.mask_CSF = mask_CSF;
option_data.N_std = N_std;
option_data.wG = [];
option_data.wG_r2p = [];
option_data.mask_FastRelax = zeros(size(r2prime_solver));
option_data.mask_SlowRelax = zeros(size(r2prime_solver));

%% Reconstruction

fprintf('Running chi_sep_MEDI...\n');

t_medi = tic;
[x_para, x_dia, x_tot] = chi_sep_MEDI( ...
    mag, ...
    local_field_hz, ...
    r2prime_solver, ...
    N_std, ...
    mask, ...
    params, ...
    option_data);
runtime_s = toc(t_medi);

%% Output validation

assert(all(isfinite(x_para(mask))), ...
    'chi_para contains non-finite values');
assert(all(isfinite(x_dia(mask))), ...
    'chi_dia contains non-finite values');
assert(all(isfinite(x_tot(mask))), ...
    'chi_tot contains non-finite values');

algebra_rmse = sqrt(mean( ...
    (x_tot(mask) - ...
    (x_para(mask) - x_dia(mask))).^2));

if algebra_rmse > 1e-8
    error( ...
        'CHI03b:AlgebraFailure', ...
        'chi_tot != chi_para - chi_dia: RMSE=%g', ...
        algebra_rmse);
end

%% Save NIfTI

ref = niftiinfo(fullfile(chi02, 'mask.nii.gz'));

ref_float = ref;
ref_float.Datatype = 'single';
ref_float.BitsPerPixel = 32;

para_path = fullfile( ...
    outdir, ...
    ['chi03b_' case_id '_chi_para_ppm.nii']);

dia_path = fullfile( ...
    outdir, ...
    ['chi03b_' case_id '_chi_dia_ppm.nii']);

tot_path = fullfile( ...
    outdir, ...
    ['chi03b_' case_id '_chi_tot_ppm.nii']);

csf_path = fullfile( ...
    outdir, ...
    ['chi03b_' case_id '_mask_CSF.nii']);

niftiwrite( ...
    single(x_para), ...
    para_path, ...
    ref_float, ...
    'Compressed', true);

niftiwrite( ...
    single(x_dia), ...
    dia_path, ...
    ref_float, ...
    'Compressed', true);

niftiwrite( ...
    single(x_tot), ...
    tot_path, ...
    ref_float, ...
    'Compressed', true);

niftiwrite( ...
    uint8(mask_CSF), ...
    csf_path, ...
    ref, ...
    'Compressed', true);

%% QC metrics

inside = mask > 0;

vp = x_para(inside);
vd = x_dia(inside);
vt = x_tot(inside);
vq = qsm(inside);

qsm_rmse = sqrt(mean((vt-vq).^2));

C = corrcoef(vt, vq);
qsm_r = C(1,2);

C = corrcoef(vp, vd);
component_r = C(1,2);

qc = struct();

qc.case_id = case_id;
qc.runtime_seconds = runtime_s;

qc.mask_voxels = nnz(mask);
qc.csf_voxels = n_csf;

qc.r2prime_nonfinite_raw_whole_volume = n_nonfinite_r2p;
qc.r2prime_nonfinite_raw_in_mask = n_nonfinite_r2p_mask;
qc.r2prime_negative_voxels_in_mask = n_negative_r2p_mask;

qc.params = struct( ...
    'Dr_Hz_per_ppm', 137, ...
    'lambda', 1, ...
    'lambda_CSF', 1, ...
    'CSF_flag_erode', csf_flag_erode, ...
    'CSF_thresh_R2s_per_s', csf_thresh_R2s, ...
    'CF_Hz', CF, ...
    'B0_T', double(p.B0), ...
    'B0_dir', params.b0_dir, ...
    'voxel_size_mm', voxel_size);

qc.chi_para = stats(vp);
qc.chi_dia = stats(vd);
qc.chi_tot = stats(vt);

qc.algebra_rmse_ppm = algebra_rmse;
qc.qsm_rmse_ppm = qsm_rmse;
qc.qsm_pearson_r = qsm_r;
qc.para_dia_pearson_r = component_r;

qc.status = 'COMPLETED';

qc_path = fullfile( ...
    outdir, ...
    ['chi03b_' case_id '_qc.json']);

fid = fopen(qc_path, 'w');

if fid < 0
    error('Could not open QC JSON for writing');
end

fprintf(fid, '%s\n', jsonencode(qc, PrettyPrint=true));
fclose(fid);

fprintf( ...
    'CHI03b completed: %s | runtime %.1f s | CSF %d voxels\n', ...
    case_id, runtime_s, n_csf);

fprintf( ...
    'chi_tot vs QSM: r=%.6f RMSE=%.6f ppm\n', ...
    qsm_r, qsm_rmse);

fprintf( ...
    'corr(chi_para,chi_dia)=%.6f\n', ...
    component_r);

end


function s = stats(v)

p = prctile(v, [0 0.1 1 5 50 95 99 99.9 100]);

s = struct();
s.min = p(1);
s.p0_1 = p(2);
s.p1 = p(3);
s.p5 = p(4);
s.median = p(5);
s.p95 = p(6);
s.p99 = p(7);
s.p99_9 = p(8);
s.max = p(9);

end
