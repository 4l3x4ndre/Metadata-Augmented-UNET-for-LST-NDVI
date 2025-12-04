import ee
import os
from dotenv import load_dotenv 

load_dotenv()


ee.Authenticate()

ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))

print(ee.String('Hello from the Earth Engine servers!').getInfo())


# Load a Landsat image.
img = ee.Image('LANDSAT/LT05/C02/T1_L2/LT05_034033_20000913')

# Print image object WITHOUT call to getInfo(); prints serialized request instructions.
print(img)

# Print image object WITH call to getInfo(); prints image metadata.
print(img.getInfo())
