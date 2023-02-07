import io
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
from PIL import Image

# # Azure Blob config
conn_str = "DefaultEndpointsProtocol=https;AccountName=preventer;AccountKey=uLzANa2nsp7FGZjbvMkztStgvCnT84vOC2dsmhpH/Z9BRG6M2ytosbiCtSNN2Em/GmsD6QhuGzeD+ASt3+Jn7w==;EndpointSuffix=core.windows.net"
container = "danger"

def send_blob(frame):

    # # connect Azure Storage blob
    service_client = BlobServiceClient.from_connection_string(conn_str)

    now = datetime.today().strftime('%Y%m%d_%H%M%S')
    file_name = f"photo_{now}.png"
    image_frame = Image.fromarray(frame)

    binary = io.BytesIO()
    image_frame.save(binary, format="png")
    png_buffer = binary.getvalue()

    print(type(png_buffer))

    # send to blob
    blob_client = service_client.get_blob_client(container=container, blob=file_name)
    blob_client.upload_blob(png_buffer, overwrite=True, content_settings="image/png")



