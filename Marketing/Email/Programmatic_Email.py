import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import xlrd
import schedule
import time
import codecs

def realtor_email_v3():

    MY_ADDRESS = 'PRIVATE'
    MY_PASSWORD = 'PRIVATE'

    with xlrd.open_workbook('Realtor_MI_Emails_Final.xlsx','r') as workbook:

        worksheet=workbook.sheet_by_index(0)

        for index in range(1, worksheet.nrows):
            try:
                # time.sleep(1)
                fromaddr = 'Wall Street Lending <apply@wslmi.com>'
                toaddr = str(worksheet.cell(index,9).value)

                msg = MIMEMultipart('alternative')
                msg['From'] = fromaddr
                msg['To'] = toaddr
                msg['Subject'] = "How do you qualify more clients?" #"DropBox HTML IMG Link" #"Base64 Embedded HTML IMG"


                with codecs.open("Realtor_Email_V1.html","r") as body4:


                    html_msg = MIMEText(body4.read(), 'html')
                    msg.attach(html_msg)

                    s = smtplib.SMTP_SSL(host='email-smtp.us-east-1.amazonaws.com', port=465)
                    s.ehlo()
                    s.login(MY_ADDRESS, MY_PASSWORD)
                    text = msg.as_string()
                    s.sendmail(fromaddr, toaddr, text)
                    # print("Successfully sent email {}".format(str(worksheet.cell(index,3).value)))
                    print "Successfully sent email"
                    print toaddr
                    print index
                    s.quit()

            except Exception:
                # print("Error: unable to send email for {}".format(str(worksheet.cell(index,3).value)))
                print "Error: unable to send email for"
                print toaddr
                print index
                pass


schedule.every().tuesday.at("10:00").do(realtor_email_v3)

while True:
    schedule.run_pending()
    time.sleep(1)



