import boto3

# Crear un cliente S3
s3 = boto3.client('s3')

# Definir el nombre del bucket y la ruta del archivo
bucket_name = 'datalatet01740327929864'
file_path = 'multipleregression/quakes.csv'

# Descargar el archivo desde S3
with open('quakes.csv', 'wb') as f:
    s3.download_fileobj(bucket_name, file_path, f)
