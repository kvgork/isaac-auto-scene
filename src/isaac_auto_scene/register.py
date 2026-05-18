"""Point-cloud registration: FPFH+FGR global then GICP local refinement.

# Phase 5 — FPFH+FGR global → GICP local; 3-5 random restarts; fitness/RMSE
#            quality gate; fallback to GeoTransformer (GPU) → TEASER++.
"""
