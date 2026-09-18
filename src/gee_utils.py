"""Google Earth Engine helpers for the EcoExplain-VLM five-source cube.

Ported from Dataset_Extraction.ipynb + dataset_fixes.py (Kanha pilot):
  1. Sentinel-2 SR: B2–B8, B8A, B11, B12 + NDVI/EVI/NDMI/NBR, SCL mask
  2. Sentinel-1 GRD: VV, VH (dominant IW orbit pass)
  3. CHIRPS daily rainfall sum (mm)
  4. ERA5-Land month-matched anomalies: temperature, soil moisture
  5. NASADEM: elevation, slope, aspect
"""

from __future__ import annotations

import datetime

import ee

S2_OPTICAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDICES = ["NDVI", "EVI", "NDMI", "NBR"]

ERA5_SCALE = 11132
CHIRPS_SCALE = 5566
DEM_SCALE = 30
S2_SCALE = 10
S1_SCALE = 10


def mask_s2_clouds(image):
    """SCL mask: drop shadow (3), cloud (8–9), cirrus (10), snow (11). Run before indices."""
    scl = image.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask)


def add_indices(image):
    """NDVI, EVI (reflectance-scaled), NDMI, NBR."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8").divide(10000),
            "RED": image.select("B4").divide(10000),
            "BLUE": image.select("B2").divide(10000),
        },
    ).rename("EVI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    return image.addBands([ndvi, evi, ndmi, nbr])


def get_s2_composite(roi, start, end, max_cloud_pct=20):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_s2_clouds)
        .map(add_indices)
        .select(S2_OPTICAL_BANDS + S2_INDICES)
    )
    return coll.median(), coll.size()


def _observation_months(obs_start, obs_end):
    start = datetime.datetime.strptime(obs_start, "%Y-%m-%d")
    end = datetime.datetime.strptime(obs_end, "%Y-%m-%d")
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        if cur.month not in months:
            months.append(cur.month)
        cur = (
            cur.replace(year=cur.year + 1, month=1)
            if cur.month == 12
            else cur.replace(month=cur.month + 1)
        )
    return months


def get_era5_month_matched_anomaly(
    roi, obs_start, obs_end, variable, clim_start="2015-01-01", clim_end="2021-12-31"
):
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    months = _observation_months(obs_start, obs_end)

    def month_anomaly(month):
        obs = (
            era5.filterDate(obs_start, obs_end)
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .select(variable)
            .mean()
        )
        clim = (
            era5.filterDate(clim_start, clim_end)
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .select(variable)
            .mean()
        )
        return obs.subtract(clim)

    anomalies = ee.ImageCollection([month_anomaly(m) for m in months])
    suffix_map = {
        "temperature_2m": "temp",
        "total_precipitation_sum": "precip",
        "volumetric_soil_water_layer_1": "soil_moisture",
    }
    suffix = suffix_map.get(variable, variable)
    return anomalies.mean().rename(f"{suffix}_anomaly")


def get_era5_anomalies(roi, obs_start, obs_end, clim_start="2015-01-01", clim_end="2021-12-31"):
    temp = get_era5_month_matched_anomaly(
        roi, obs_start, obs_end, "temperature_2m", clim_start, clim_end
    )
    soil_moisture = get_era5_month_matched_anomaly(
        roi, obs_start, obs_end, "volumetric_soil_water_layer_1", clim_start, clim_end
    )
    return temp, soil_moisture


def get_chirps_rainfall(roi, start, end):
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(roi)
        .filterDate(start, end)
        .select("precipitation")
    )
    total = chirps.reduce(ee.Reducer.sum()).rename("rainfall_mm")
    return total, chirps.size()


def get_dem_terrain(roi):
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").clip(roi)
    terrain = ee.Terrain.products(dem)
    return ee.Image.cat(
        [
            terrain.select("elevation").rename("elevation_m"),
            terrain.select("slope").rename("slope_deg"),
            terrain.select("aspect").rename("aspect_deg"),
        ]
    )


def get_dominant_orbit_pass(roi, start, end):
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filterDate(start, end)
    )
    hist = coll.aggregate_histogram("orbitProperties_pass").getInfo()
    if not hist:
        return "ASCENDING"
    return max(hist, key=hist.get)


def add_s1_metrics(image):
    vv_db = image.select("VV")
    vh_db = image.select("VH")
    vv_linear = ee.Image(10).pow(vv_db.divide(10))
    vh_linear = ee.Image(10).pow(vh_db.divide(10))
    vv_vh_ratio = vv_linear.divide(vh_linear).rename("VV_VH_ratio")
    return image.addBands(vv_vh_ratio)


def get_s1_composite(roi, start, end, orbit_pass=None):
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filterDate(start, end)
    )
    if orbit_pass:
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    coll = coll.select(["VV", "VH"]).map(add_s1_metrics)
    return coll.median(), coll.size()


def reduce_coarse(img, lon, lat, band, scale):
    """Buffer the patch centroid to the native coarse resolution (ERA5 / CHIRPS)."""
    center = ee.Geometry.Point([lon, lat])
    buffered = center.buffer(scale).bounds()
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffered,
        scale=scale,
        maxPixels=1e8,
    ).get(band)


def reduce_fine(img, geom, band, scale=30):
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e8,
    ).get(band)


def patch_geometry(lon, lat, patch_size_m, roi):
    raw = ee.Geometry.Point([lon, lat]).buffer(patch_size_m / 2).bounds()
    return raw.intersection(roi, ee.ErrorMargin(1))


def masked_fraction(image, geom, scale=10):
    unmasked = image.select(0).mask()
    stats = unmasked.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9
    )
    frac_present = ee.Number(stats.values().get(0))
    return ee.Number(1).subtract(frac_present)


def rgb_reflectance(s2_image):
    return s2_image.select(["B4", "B3", "B2"]).divide(10000).rename(["R", "G", "B"])
