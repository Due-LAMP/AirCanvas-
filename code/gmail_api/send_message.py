import base64
import os
import mimetypes
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from io import BytesIO

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

# Gmail API 스코프
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


def get_credentials():
  """OAuth 2.0 인증을 통해 credentials 가져오기"""
  creds = None
  _dir = os.path.dirname(os.path.abspath(__file__))
  token_path = os.path.join(_dir, '../token.json')
  creds_path = os.path.join(_dir, '../credentials.json')
  
  # token.json에 저장된 인증 정보 로드
  if os.path.exists(token_path):
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
  
  # 유효한 자격 증명이 없으면 로그인
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      if not os.path.exists(creds_path):
        print(f"❌ {creds_path} 파일을 찾을 수 없습니다.")
        return None
      
      flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
      creds = flow.run_local_server(port=0)
    
    # 자격 증명 저장
    with open(token_path, 'w') as token:
      token.write(creds.to_json())
  
  return creds


def gmail_send_message_with_attachment(attachment_filename):
  """첨부파일이 있는 이메일 전송 (uploadType=media 사용)
  Returns: Message object, including message id
  """
  creds = get_credentials()
  if not creds:
    return None
  
  try:
    service = build("gmail", "v1", credentials=creds)
    
    # MIME 메시지 생성
    message = MIMEMultipart()
    message["To"] = "tlsdbfk0000@gmail.com"
    message["From"] = "jhkmo51@gmail.com"
    message["Subject"] = "📸 Your 4-Cut Photo is Here!"
    
    # 본문 추가
    body = MIMEText("""Hello!

Here are your amazing 4-cut photobooth pictures! 📷✨

Thank you for using our photobooth!

Best regards,
4-Cut Photobooth Team
""")
    message.attach(body)
    
    # 첨부파일 추가
    if os.path.exists(attachment_filename):
      content_type, encoding = mimetypes.guess_type(attachment_filename)
      
      if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
      
      main_type, sub_type = content_type.split('/', 1)
      
      with open(attachment_filename, 'rb') as fp:
        if main_type == 'image':
          img = MIMEImage(fp.read(), _subtype=sub_type)
        else:
          img = MIMEBase(main_type, sub_type)
          img.set_payload(fp.read())
      
      filename = os.path.basename(attachment_filename)
      img.add_header('Content-Disposition', 'attachment', filename=filename)
      message.attach(img)
    else:
      print(f"❌ 첨부파일을 찾을 수 없습니다: {attachment_filename}")
      return None
    
    # RFC822 형식으로 변환
    raw_message = message.as_bytes()
    
    # BytesIO로 메모리 스트림 생성
    media = MediaIoBaseUpload(
      BytesIO(raw_message),
      mimetype='message/rfc822',
      resumable=True
    )
    
    # uploadType=media를 사용한 간단한 업로드
    send_message = (
        service.users()
        .messages()
        .send(
          userId="me",
          media_body=media,
          uploadType='media'
        )
        .execute()
    )
    
    print(f'✅ Message Id: {send_message["id"]}')
    print(f'✅ 이메일 전송 성공!')
    
  except HttpError as error:
    print(f"❌ An error occurred: {error}")
    send_message = None
  except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {attachment_filename}")
    send_message = None
  
  return send_message


def gmail_send_message():
  """Create and send an email message
  Print the returned  message id
  Returns: Message object, including message id
  """
  creds = get_credentials()
  if not creds:
    return None

  try:
    service = build("gmail", "v1", credentials=creds)
    message = EmailMessage()

    message.set_content("This is automated mail from Photobooth")

    message["To"] = "tlsdbfk0000@gmail.com"
    message["From"] = "jhkmo51@gmail.com"
    message["Subject"] = "Test message from Photobooth"

    # encoded message
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {"raw": encoded_message}
    # pylint: disable=E1101
    send_message = (
        service.users()
        .messages()
        .send(userId="me", body=create_message)
        .execute()
    )
    print(f'Message Id: {send_message["id"]}')
  except HttpError as error:
    print(f"An error occurred: {error}")
    send_message = None
  return send_message


if __name__ == "__main__":
  # 첨부파일 있는 버전 실행
  gmail_send_message_with_attachment(
    attachment_filename = "/home/willtek/work/AirCanvas-/code/photobooth_output/20260415_141954/4cut.jpg"

  )