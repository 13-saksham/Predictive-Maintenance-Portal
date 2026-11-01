"""API + website. Run: uvicorn app:app --port 8000"""
import json, time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

A = json.loads(Path("artifacts.json").read_text())
M = {m["machineId"]: m for m in A["machines"]}
WF = Path("workorders.json")
WO = json.loads(WF.read_text()) if WF.exists() else []
now = lambda: time.strftime("%Y-%m-%d %H:%M")
save = lambda: WF.write_text(json.dumps(WO, indent=1))
ACT = {  # generic inspection steps per subsystem; no measurements are invented
    "Fan": ["Inspect fan blades for damage, erosion or imbalance", "Review fan speed and bypass ratio trends", "Check fan case and inlet for foreign object damage"],
    "LPC": ["Borescope inspection of the low-pressure compressor", "Review LPC outlet temperature trend", "Check variable guide vanes and bleed valves"],
    "HPC": ["Borescope inspection of HPC blades and stators", "Review compressor efficiency and pressure trends", "Check variable stator vane and bleed valve operation"],
    "Combustor": ["Borescope inspection of the combustor and fuel nozzles", "Review fuel flow against pressure trend", "Check igniters and fuel system filters"],
    "HPT": ["Borescope inspection of HPT blades and nozzles", "Check turbine cooling air flow", "Review turbine temperature margin trend"],
    "LPT": ["Borescope inspection of LPT blades and nozzles", "Check turbine cooling air flow", "Review LPT outlet temperature trend"]}
app = FastAPI()
wo_of = lambda mid: next((w["id"] for w in WO if w["machineId"] == mid and w["status"] != "Cancelled"), None)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/api/machines")
def machines(band: Optional[str] = None):
    keep = ("machineId", "dataset", "cyclesObserved", "predictedRul", "rulLower", "rulUpper", "failureProb30", "riskScore", "band")
    out = [dict({k: m[k] for k in keep}, componentName=m["component"]["name"], workOrderId=wo_of(m["machineId"]))
           for m in A["machines"] if not band or m["band"] == band]
    return sorted(out, key=lambda m: -m["riskScore"])


@app.get("/api/machines/{mid}")
def machine(mid: str):
    if mid not in M:
        raise HTTPException(404, "Machine not found")
    return dict(M[mid], workOrderId=wo_of(mid))


@app.get("/api/model")
def model():
    return A["model"]


class New(BaseModel):
    machineId: str


class Patch(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/work-orders")
def create(b: New):
    m = M.get(b.machineId)
    if not m:
        raise HTTPException(404, "Machine not found")
    if wo_of(b.machineId):  # idempotent: one live order per machine
        return next(w for w in WO if w["id"] == wo_of(b.machineId))
    c = m["component"]["name"]
    w = dict(id=f"WO-{len(WO) + 1:04d}", machineId=b.machineId, status="Draft", priority=m["band"],
             title=f"Inspect {c} on {b.machineId}",
             summary=f"Predicted RUL is {m['predictedRul']:.0f} cycles (range {m['rulLower']:.0f} to {m['rulUpper']:.0f}), "
                     f"with a {m['failureProb30'] * 100:.0f}% chance of failure within 30 cycles. Likely component (indicative): {c}.",
             evidence=dict(predictedRul=m["predictedRul"], rulLower=m["rulLower"], rulUpper=m["rulUpper"],
                           failureProb30=m["failureProb30"], cyclesObserved=m["cyclesObserved"]),
             likelyComponent=m["component"], recommendedActions=ACT[c],
             latestInterventionCycles=int(0.7 * m["rulLower"]), notes="", createdAt=now(), updatedAt=now())
    WO.append(w)
    save()
    return w


@app.get("/api/work-orders")
def orders():
    return WO[::-1]


@app.get("/api/work-orders/{wid}")
def order(wid: str):
    w = next((w for w in WO if w["id"] == wid), None)
    if not w:
        raise HTTPException(404, "Work order not found")
    return w


@app.patch("/api/work-orders/{wid}")
def patch(wid: str, b: Patch):
    w = order(wid)
    if w["status"] != "Draft":
        raise HTTPException(409, "Only Draft work orders can be changed")
    if b.status:
        if b.status not in ("Approved", "Cancelled"):
            raise HTTPException(422, "Status must be Approved or Cancelled")
        w["status"] = b.status
    if b.notes is not None:
        w["notes"] = b.notes
    w["updatedAt"] = now()
    save()
    return w
