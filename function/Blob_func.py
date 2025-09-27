import io
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from PIL import Image

# # Azure Blob config
conn_str = ""
container = "danger"

def send_blob(frame):

    # # connect Azure Storage blob
    service_client = BlobServiceClient.from_connection_string(conn_str)

    now = datetime.today().strftime('%Y%m%d_%H%M%S')
    file_name = f"photo_{now}.png"
    image_frame = Image.fromarray(frame)
    img_byte_arr = io.BytesIO()
    image_frame.save(img_byte_arr, format="png")
    png_buffer = img_byte_arr.getvalue()

    # send to blob
    blob_client = service_client.get_blob_client(container=container, blob=file_name)
    blob_client.upload_blob(png_buffer, overwrite=True)



