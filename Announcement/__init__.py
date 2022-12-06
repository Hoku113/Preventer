import os
import logging
import azure.functions as func
import smtplib
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.utils import formatdate
from email.mime.multipart import MIMEMultipart

def main(myblob: func.InputStream):

    # Get Host Gmail address and Password
    HOST_ACCOUNT = os.environ["Account"]
    HOST_PASSWORD = os.environ["Password"]

    # Get destination Mail address and Subject and text
    SUBJECT = os.environ['Subject']
    TO_ACCOUNT = os.environ['To']
    TEXT = os.environ['Text']

    # Connecting SMTP Server
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(HOST_ACCOUNT, HOST_PASSWORD)

    # Create Mail

    msg = MIMEMultipart()
    msg['Subject'] = SUBJECT
    msg['From'] = HOST_ACCOUNT
    msg['To'] = TO_ACCOUNT
    msg['Date'] = formatdate(localtime=True)

    # Create message text
    body = MIMEText(TEXT)
    msg.attach(body)

    # Create image 
    img = myblob.read()
    img = MIMEImage(img, name=myblob.name)
    msg.attach(img)

    # send mail
    smtpobj.send_message(msg)
    smtpobj.close()