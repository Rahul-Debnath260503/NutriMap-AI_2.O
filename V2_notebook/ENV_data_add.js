// =====================================================================
// 1. LOAD CSV ASSET AND REBUILD GEOMETRY
// =====================================================================

var soil_raw = ee.FeatureCollection(
  'projects/ee-rahulds24/assets/Kerala_Soil_ML_Ready'
);

var soil = soil_raw.map(function(f){

  var lat = ee.Number(f.get('Latitude'));
  var lon = ee.Number(f.get('Longitude'));

  return ee.Feature(
    ee.Geometry.Point([lon, lat]),
    f.toDictionary()
  );

});

print('Total Samples', soil.size());

Map.centerObject(soil, 8);

Map.addLayer(
  soil,
  {color:'red'},
  'Soil Samples'
);


// =====================================================================
// 2. TERRAIN VARIABLES
// =====================================================================

var srtm = ee.Image('USGS/SRTMGL1_003');

var elevation = srtm.select('elevation')
                    .rename('Elevation');

var terrain = ee.Terrain.products(srtm);

var slope = terrain.select('slope')
                   .rename('Slope');

var aspect = terrain.select('aspect')
                    .rename('Aspect');


// =====================================================================
// 3. SENTINEL-2 CLOUD MASK
// =====================================================================

function maskS2(image) {

  var scl = image.select('SCL');

  var mask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))      // cloud medium prob
    .and(scl.neq(9))      // cloud high prob
    .and(scl.neq(10))     // cirrus
    .and(scl.neq(11));    // snow

  return image.updateMask(mask);
}

var s2 = ee.ImageCollection(
    'COPERNICUS/S2_SR_HARMONIZED'
)
.filterBounds(soil)
.filterDate('2023-01-01', '2023-12-31')
.map(maskS2)
.median();


// =====================================================================
// 4. NDVI
// =====================================================================

var ndvi = s2.normalizedDifference(
    ['B8','B4']
).rename('NDVI');


// =====================================================================
// 5. NDWI
// =====================================================================

var ndwi = s2.normalizedDifference(
    ['B3','B8']
).rename('NDWI');


// =====================================================================
// 6. RAINFALL (ANNUAL TOTAL)
// =====================================================================

var rainfall = ee.ImageCollection(
    'UCSB-CHG/CHIRPS/DAILY'
)
.filterDate('2023-01-01','2023-12-31')
.sum()
.rename('Rainfall');


// =====================================================================
// 7. TEMPERATURE (ANNUAL MEAN °C)
// =====================================================================

var temperature = ee.ImageCollection(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR'
)
.filterDate('2023-01-01','2023-12-31')
.select('temperature_2m')
.mean()
.subtract(273.15)
.rename('Temperature_C');


// =====================================================================
// 8. SOIL MOISTURE
// ERA5 version (more stable than SMAP)
// =====================================================================

var soilMoisture = ee.ImageCollection(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR'
)
.filterDate('2023-01-01','2023-12-31')
.select('volumetric_soil_water_layer_1')
.mean()
.rename('Soil_Moisture');


// =====================================================================
// 9. ESA WORLDCOVER
// =====================================================================

var lulc = ee.Image(
    'ESA/WorldCover/v200/2021'
)
.select('Map')
.rename('LULC');


// =====================================================================
// 10. SOILGRIDS
// RUN THESE FIRST TO VERIFY EXISTENCE
// =====================================================================

// Uncomment and test individually

/*
var clay = ee.Image('projects/soilgrids-isric/clay_mean')
              .select(0)
              .rename('Clay');

var sand = ee.Image('projects/soilgrids-isric/sand_mean')
              .select(0)
              .rename('Sand');

var silt = ee.Image('projects/soilgrids-isric/silt_mean')
              .select(0)
              .rename('Silt');
*/


// =====================================================================
// 11. STACK ENVIRONMENTAL VARIABLES
// =====================================================================

var envStack = ee.Image.cat([

    elevation,
    slope,
    aspect,

    ndvi,
    ndwi,

    rainfall,

    temperature,

    soilMoisture,

    lulc

]);

print('Environmental Stack', envStack);


// =====================================================================
// 12. EXTRACT VALUES TO POINTS
// =====================================================================

var enriched = envStack.sampleRegions({

    collection: soil,

    scale: 30,

    geometries: true

});

print('Sample Output');
print(enriched.limit(5));


// =====================================================================
// 13. VISUALIZATION
// =====================================================================

Map.addLayer(
    ndvi,
    {min:0,max:1},
    'NDVI'
);

Map.addLayer(
    elevation,
    {min:0,max:1500},
    'Elevation'
);

Map.addLayer(
    rainfall,
    {min:1000,max:5000},
    'Rainfall'
);


// =====================================================================
// 14. EXPORT
// =====================================================================

Export.table.toDrive({

    collection: enriched,

    description: 'Kerala_Soil_Environmental_Features',

    fileFormat: 'CSV'

});