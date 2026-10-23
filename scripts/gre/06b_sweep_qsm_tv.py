#!/usr/bin/env python3
"""GRE06b lambda sweep driver. Runs reconstructions and summarizes QC."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from datetime import datetime
from pathlib import Path

DEFAULT=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]

def tag(v): return f"{v:.8g}".replace("-","m").replace("+","").replace(".","p")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--lambdas",type=float,nargs="+",default=DEFAULT)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--max-iterations",type=int,default=500)
    p.add_argument("--min-iterations",type=int,default=50)
    p.add_argument("--check-every",type=int,default=10)
    p.add_argument("--relative-update-tol",type=float,default=1e-4)
    p.add_argument("--power-iterations",type=int,default=20)
    p.add_argument("--step-safety",type=float,default=0.95)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--reconstruction-script",type=Path,default=Path("scripts/gre/06b_reconstruct_qsm_tv.py"))
    a=p.parse_args()
    root=a.root.resolve(); recon=a.reconstruction_script
    if not recon.is_file(): raise FileNotFoundError(recon)
    lambdas=[]
    for v in a.lambdas:
        if v<=0: raise ValueError("All lambdas must be >0")
        if v not in lambdas: lambdas.append(v)
    for i,lam in enumerate(lambdas,1):
        print("="*80); print(f"[{i}/{len(lambdas)}] lambda_tv={lam:.8g}"); print("="*80)
        cmd=[sys.executable,str(recon),a.case,"--lambda-tv",str(lam),"--root",str(root),"--max-iterations",str(a.max_iterations),"--min-iterations",str(a.min_iterations),"--check-every",str(a.check_every),"--relative-update-tol",str(a.relative_update_tol),"--power-iterations",str(a.power_iterations),"--step-safety",str(a.step_safety),"--device",a.device]
        if a.overwrite: cmd.append("--overwrite")
        subprocess.run(cmd,check=True)
    rows=[]; base=root/"derivatives"/a.case/"GRE"/"06b_qsm_tv"
    for lam in lambdas:
        q=base/f"lambda_{tag(lam)}"/f"{a.case}_GRE06b_qsm_tv_qc.json"
        with open(q,"r",encoding="utf-8") as f: qc=json.load(f)
        m=qc["final_metrics"]; s=qc["solver"]
        rows.append({"lambda_tv":lam,"iterations":s["iterations_completed"],"converged":s["converged"],"final_relative_update":s["final_relative_update"],"objective":m["objective"],"data_term":m["data_term"],"residual_l2":m["residual_l2"],"residual_rmse_ppm":m["residual_rmse_ppm"],"tv_norm":m["tv_norm"],"chi_min_ppm":m["chi_min_ppm"],"chi_median_ppm":m["chi_median_ppm"],"chi_max_ppm":m["chi_max_ppm"],"chi_sd_ppm":m["chi_sd_ppm"],"qc_json":str(q.relative_to(root))})
    out=base/"lambda_sweep"; out.mkdir(parents=True,exist_ok=True)
    cp=out/f"{a.case}_GRE06b_lambda_sweep.csv"; jp=out/f"{a.case}_GRE06b_lambda_sweep.json"
    with open(cp,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(jp,"w",encoding="utf-8") as f:
        json.dump({"created":datetime.now().isoformat(timespec="seconds"),"case":a.case,"purpose":"GRE06b TV regularization sweep","lambda_selection":"No automatic selection: inspect log residual vs log TV together with QSM/residual QC.","rows":rows},f,indent=2); f.write("\n")
    print("\nSWEEP COMPLETE")
    print(f"CSV:  {cp.relative_to(root)}")
    print(f"JSON: {jp.relative_to(root)}")
    print(f"{'lambda':>12} {'iter':>6} {'conv':>5} {'RMSE ppm':>12} {'TV norm':>14} {'chi SD':>12}")
    print("-"*70)
    for r in rows:
        print(f"{r['lambda_tv']:12.5g} {r['iterations']:6d} {str(r['converged']):>5} {r['residual_rmse_ppm']:12.5e} {r['tv_norm']:14.5e} {r['chi_sd_ppm']:12.5e}")

if __name__=="__main__": main()
