import base64
import mimetypes
import os
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API 스코프
SCOPES = ['https://www.googleapis.com/auth/gmail.compose']


def get_credentials():
  """OAuth 2.0 인증을 통해 credentials 가져오기"""
  creds = None
  token_path = '../token.json'
  creds_path = '../credentials.json'
  
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
        print("Google Cloud Console에서 OAuth 클라이언트 ID를 생성하고 다운로드하세요.")
        return None
      
      flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
      creds = flow.run_local_server(port=0)
    
    # 자격 증명 저장
    with open(token_path, 'w') as token:
      token.write(creds.to_json())
  
  return creds


def gmail_create_draft_with_attachment():
  """Create and insert a draft email with attachment.
   Print the returned draft's message and id.
  Returns: Draft object, including draft id and message meta data.
  """
  creds = get_credentials()
  if not creds:
    return None

  try:
    # create gmail api client
    service = build("gmail", "v1", credentials=creds)
    mime_message = EmailMessage()

    # headers
    mime_message["To"] = "jhkmo51@gmail.com"
    mime_message["From"] = "jhkmo51@gmail.com"
    mime_message["Subject"] = "sample with attachment"

    # text
    mime_message.set_content(
        "Hi, this is automated mail with attachment.Please do not reply."
    )

    # attachment
    attachment_filename = "/home/willtek/work/AirCanvas-/code/photobooth_output/20260415_141954/4cut.jpg"
    # guessing the MIME type
    type_subtype, _ = mimetypes.guess_type(attachment_filename)
    maintype, subtype = type_subtype.split("/")

    with open(attachment_filename, "rb") as fp:
      attachment_data = fp.read()
    mime_message.add_attachment(attachment_data, maintype, subtype)

    encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

    create_draft_request_body = {"message": {"raw": encoded_message}}
    # pylint: disable=E1101
    draft = (
        service.users()
        .drafts()
        .create(userId="me", body=create_draft_request_body)
        .execute()
    )
    print(f'Draft id: {draft["id"]}\nDraft message: {draft["message"]}')
  except HttpError as error:
    print(f"An error occurred: {error}")
    draft = None
  return draft


def build_file_part(file):
  """Creates a MIME part for a file.

  Args:
    file: The path to the file to be attached.

  Returns:
    A MIME part that can be attached to a message.
  """
  content_type, encoding = mimetypes.guess_type(file)

  if content_type is None or encoding is not None:
    content_type = "application/octet-stream"
  main_type, sub_type = content_type.split("/", 1)
  if main_type == "text":
    with open(file, "rb"):
      msg = MIMEText("r", _subtype=sub_type)
  elif main_type == "image":
    with open(file, "rb"):
      msg = MIMEImage("r", _subtype=sub_type)
  elif main_type == "audio":
    with open(file, "rb"):
      msg = MIMEAudio("r", _subtype=sub_type)
  else:
    with open(file, "rb"):
      msg = MIMEBase(main_type, sub_type)
      msg.set_payload(file.read())
  filename = os.path.basename(file)
  msg.add_header("Content-Disposition", "attachment", filename=filename)
  return msg


if __name__ == "__main__":
  gmail_create_draft_with_attachment()