"""Truth appears only in evaluation, never in estimator admission decisions."""
import numpy as np


def score(estimate, truth):
    result={"accepted":estimate.accepted,"reasons":estimate.reasons,"runtime_ms":estimate.elapsed_ms,
            "residual":estimate.residual,"condition":estimate.condition,
            "center_mm":None,"normal_deg":None,"diameter_mm":None,"truth_pass":False,
            "normal_component_mm":None,"tangent_mm":None,"xyz_mm":None,"coverage95":None}
    if estimate.center is None or estimate.normal is None or estimate.radius is None:
        return result
    n=np.array(truth["normal"],float)
    if n[2]<0:
        n=-n
    delta=(estimate.center-np.array(truth["center_m"]))*1000
    result.update(center_mm=float(np.linalg.norm(delta)),xyz_mm=delta.tolist(),
                  normal_component_mm=float(delta@n),tangent_mm=float(np.linalg.norm(delta-(delta@n)*n)),
                  normal_deg=float(np.rad2deg(np.arccos(np.clip(estimate.normal@n,-1,1)))),
                  diameter_mm=float(2*abs(estimate.radius-truth["radius_m"])*1000),
                  estimate={"center_m":estimate.center.tolist(),"normal":estimate.normal.tolist(),"radius_m":float(estimate.radius)},
                  diagnostics=estimate.diagnostics)
    result["truth_pass"]=bool(result["center_mm"]<=4 and result["normal_deg"]<=2 and result["diameter_mm"]<=2)
    cov=estimate.covariance_center
    if cov is not None and np.all(np.isfinite(cov)) and np.min(np.linalg.eigvalsh(cov))>0:
        d=delta/1000
        result["coverage95"]=bool(d@np.linalg.solve(cov,d)<=7.814727903251179)
        result["center_std_mm"]=(1000*np.sqrt(np.diag(cov))).tolist()
    return result


def summarize(records):
    groups={}
    for row in records:
        key=tuple(row[k] for k in ("split","geometry","evidence","factor","method"))
        groups.setdefault(key,[]).append(row)
    output=[]
    for key,rows in sorted(groups.items()):
        n=len(rows)
        accepted=[r for r in rows if r["accepted"]]
        r=dict(zip(("split","geometry","evidence","factor","method"),key))
        r.update(n=n,solved_n=sum(x["center_mm"] is not None for x in rows),accepted=len(accepted),success_rate=sum(x["accepted"] and x["truth_pass"] for x in rows)/n,
                 rejection_rate=1-len(accepted)/n,
                 false_accept_rate=sum(x["accepted"] and not x["truth_pass"] for x in rows)/n,
                 false_among_accepted=(sum(not x["truth_pass"] for x in accepted)/len(accepted) if accepted else None))
        for metric in ("center_mm","normal_deg","diameter_mm","runtime_ms","tangent_mm","normal_component_mm"):
            # Preserve errors of rejected but solved cases, no success-only filtering.
            values=[abs(x[metric]) for x in rows if x[metric] is not None]
            for suffix,percent in (("median",50),("p95",95),("max",100)):
                r[f"{metric}_{suffix}"]=float(np.percentile(values,percent)) if values else None
        cover=[x["coverage95"] for x in rows if x["coverage95"] is not None]
        r["conditional_coverage95"]=float(np.mean(cover)) if cover else None
        r["conditional_coverage_n"]=len(cover)
        output.append(r)
    return output
