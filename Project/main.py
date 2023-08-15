import psycopg2
from datetime import datetime, date
import random
import smtplib
import psycopg2.extras
import tensorflow as tf
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./models/object_detection.h5')
arr = [200]
arr1 = [200]
image = cv2.imread('test2.jpg')
image = imutils.resize(image, width=300)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray_image, 30, 200)
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)

i = 1
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx

    x, y, w, h = cv2.boundingRect(c)
    new_img = image[y:y+h, x:x+w]
    cv2.imwrite('./'+str(i)+'.png', new_img)
    i += 1
    break


def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            #  Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')

    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


img = cv2.imread('1.png')
char = segment_characters(img)


def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        img_ = cv2.resize(ch, (28, 28))
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_ = model.predict(img)[0]  # predicting the class

        indice = np.where(y_ == np.amax(y_))
        dic[indice[0][0]]
        character = dic[indice[0][0]]
        print(character[5])
        output.append(character)  # storing the result in a list
    plate_number = ''.join(output)

    return plate_number

b = show_results().lower()

hostname= 'localhost'
database = 'trafficdb'
username = 'postgres'
pwd = 'traffic'
port_id = 5432
conn = None
cur = None


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
date = date.today()
speed = random.randint(90,150)
try:
        conn = psycopg2.connect(
                host = hostname,
                dbname = database,
                user = username,
                password = pwd,
                port = port_id)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)


       
        cur.execute(f"SELECT * FROM vehicles WHERE vehicleid = '{b}' ")
        for record in cur.fetchall():
            arr = record

        insert_script1 = 'INSERT INTO overspeed(vehicleid,ownerName,location,time,date ,speed) VALUES (%s, %s, %s, %s, %s, %s)'
        insert_values1 = [
            (arr[0], arr[1], 'Ernakulam-Edapally', current_time,date,speed)
        ]


        for record1 in insert_values1:
            cur.execute(insert_script1, record1)

        cur.execute("SELECT * FROM overspeed")
        for record in cur.fetchall():
            arr1 = record

        conn.commit()
except Exception as error:
        print(error)
finally:
        if cur is not None:
                cur.close()
        if conn is not None:
                conn.close()


server = smtplib.SMTP('smtp.gmail.com', 587)

server.starttls()





gmail_user = 'smarttraffictracker@gmail.com'
gmail_password = 'jzefzokjpthlmfcs'

sent_from = gmail_user
to = ['smarttraffictracker@gmail.com', arr[2]]
subject = 'Over speed detection'
body = 'hey %s your %s with registration number %s has detected over speeding at %s crossing speed limit of 90km/hr and the detected speed is %s on %s %s'%(arr[1], arr[4], arr[0], arr1[2], arr1[5], arr1[4], arr1[3])

email_text = """\
From: %s
To: %s
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail(sent_from, to, email_text)
    server.close()

    print('Email sent!')
except:
    print('Something went wrong...')


with open('views.py', 'r+') as file:
   content = file.read()
   file.seek(0)
   new_line = 'resultText = "'+body+'"\n'
   file.write(new_line + content)



