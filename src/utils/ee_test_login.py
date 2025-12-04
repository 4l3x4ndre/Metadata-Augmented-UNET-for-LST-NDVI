import ee
import dotenv
import os

dotenv.load_dotenv()

service_account = os.getenv('GEE_SERVICE_ACCOUNT')
credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
ee.Initialize(credentials)
print(ee.String('Hello from the Earth Engine servers!').getInfo())
