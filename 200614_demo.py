# -*-encoding:utf-8-*-
import RPi.GPIO as GPIO  # RPi.GPIO에 정의된 기능을 GPIO라는 명칭으로 사용
import time
import picamera  # time 모듈
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pytesseract
from PIL import Image
import pymysql
import time
from gpiozero import LED
from time import sleep
from twilio.rest import Client
import pygame

#
led = LED(27)
GPIO.setmode(GPIO.BCM)  # GPIO 이름은 BCM 명칭 사용
GPIO.setup(17, GPIO.OUT)  # Trig=17 초음파 신호 전송핀 번호 지정 및 출력지정
GPIO.setup(18, GPIO.IN)  # Echo=18 초음파 수신하는 수신 핀 번호 지정 및 입력지정
camera = picamera.PiCamera()
print("start")
isParked = 0  ### 차 주차되어있는지 확인 0: 없을 때/ 1: 장애인차량 주차/ 2: 불법차량 주차
sendSMS = 0
DISTANCE = 15


def imgProcessing(distance):
    plt.style.use('dark_background')

    # read input image
    original = cv2.imread('auto-picture-' + str(distance) + 'cm.jpg')
    cv2.imshow("dd", original)
    cv2.waitKey(0)
    height, width, channel = original.shape

    # plt.figure('original',figsize=(12,10))
    # plt.imshow(original, cmap='gray')
    # plt.show()

    # convert image to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # plt.figure('gray',figsize=(12,10))
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    #
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # gaussian blur and adaptive thresholding
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # plt.figure('threshold',figsize=(6,5))
    # plt.imshow(img_thresh, cmap='gray')
    # plt.show()

    # find contours
    contours, hieararchy = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    # plt.figure('contour',figsize=(6,5))
    # plt.imshow(temp_result)
    # plt.show()

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    # prepare data
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # plt.figure('boundingrect',figsize=(12,10))
    # plt.imshow(temp_result, cmap='gray')
    # plt.show()

    # select candidates by char size
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    # bounding rect_2
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt  # index save
            cnt += 1
            possible_contours.append(d)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    # plt.figure('boundingrect_2',figsize=(12,10))
    # plt.imshow(temp_result,cmap='gray')
    # plt.show()

    # bounding rect_3
    # select candidates by arrangement of contours
    MAX_DIAG_MULTIPLYER = 5  # distance of retagles
    MAX_ANGLE_DIFF = 12.0  #
    MAX_AREA_DIFF = 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

                # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            # recursive
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break
        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    # boundingrect_3
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            # cv2. drawContours(temp_result, d['contour'], -1, (255,255,255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255), thickness=2)

    # plt.figure('boundingrect_3', figsize=(12,10))
    # plt.imshow(temp_result, cmap='gray')
    # plt.show()

    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    # affine
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        # plt.figure('img_cropeed',figsize=(6,5))
        # plt.imshow(img_cropped, cmap='gray')
        # plt.show()

    longest_idx, longest_text = -1, 0
    plate_chars = []
    # thresholding(cropped_img)
    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=(cv2.THRESH_BINARY | cv2.THRESH_OTSU))

        # find contours again
        contours, hierarchy = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)

        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #     pytesseract.pytesseract.tesseract_cmd='/usr/local/share/tessdata/'
        chars = pytesseract.image_to_string(img_result, lang='kor')
        #     print(chars)
        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

    #     plt.sublpot(len(plate_imgs),1, i+1)
    #     plt.imshow(imt_result, cmap='gray')
    # plt.show()
    # result
    info = plate_infos[longest_idx]
    chars = plate_chars[longest_idx]
    img_out = original.copy()
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                  color=(255, 0, 0), thickness=2)
    cv2.imwrite(chars + '.jpg', img_out)

    # plt.figure(figsize=(6,5))
    # plt.imshow(img_out)
    # plt.show()
    # a = "52가3108"
    # print(chars == a)
    # print(''.join(list(chars)) == "52가3108")
    # chars = ''.join(x for x in chars)
    # print(chars)

    # ## 추출되는 번호판 배열도 저장
    # exam=str(chars)
    exam = chars
    print(exam)
    return exam
    # exam = ''.join(exam.split())
    # # # ## 두개의 배열 비교해서 일치하면 count 해서 글자수로 나누고 100 곱해서 퍼센트 출력
    #
    #


def check(a, b):
    isTrue = False
    for i, j in zip(a, b):
        if i.isdigit():
            isTrue = int(i) == int(j)
            if isTrue is False:
                return isTrue
        else:
            isTrue = i == j
            if isTrue is False:
                return isTrue

    return isTrue


def checkDB(exam):
    count = 0

    music_file = "/home/pi/Desktop/using_folder/alertsound2.mp3"

    try:
        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='passwd', db='alramy_db', charset='utf8')
        pygame.mixer.init()
        pygame.mixer.music.load(music_file)

        cur = conn.cursor()
        cur.execute("select count(*) from disabled_number")
        idx = cur.fetchall()

        for i in range(1, idx[0][0] + 1):
            cur.execute("select car_number from disabled_number where idx=%s", i)
            car_num = cur.fetchone();
            car_num = str(car_num[0])

            if check(car_num, exam) is False:
                count += 1
                continue
            else:
                print("장애인 차량 ")
                return 1

        # print(count)
        if count == idx[0][0]:
            print("불법차량 ")
            pygame.mixer.music.play()
            cnt = 0
            while cnt <= 14:
                led.on()
                sleep(0.5)
                led.off()
                sleep(0.5)
                cnt += 1
            #         time.sleep(10)
            return 2
    finally:
        cur.close()
        conn.close()


try:
    while True:
        GPIO.output(17, False)
        time.sleep(0.5)

        GPIO.output(17, True)  # 10us 펄스를 내보낸다.
        time.sleep(0.00001)  # Python에서 이 펄스는 실제 100us 근처가 될 것이다
        GPIO.output(17, False)  # 하지만 HC-SR04 센서는 이 오차를 받아준다

        while GPIO.input(18) == 0:  # 18번 핀이 OFF 되는 시점을 시작 시간으로 잡는다
            start = time.time()

        while GPIO.input(18) == 1:  # 18번 핀이 다시 ON 되는 시점을 반사파 수신시간으로 잡는다
            stop = time.time()

        time_interval = stop - start  # 초음파가 수신되는 시간으로 거리를 계산한다
        distance = time_interval * 17000
        distance = round(distance, 2)

        print('Distance => ', distance, 'cm')
        if distance < DISTANCE and isParked == 0:  ### 주차된 차 없을 때, 차가 주차구역 진입시
            print("Take a picture")
            camera.capture('auto-picture-' + str(distance) + 'cm.jpg')

            exam = imgProcessing(distance)
            isParked = checkDB(exam)  ### 사진찍기
            time.sleep(30)  ### 차 빼는 시간 기다려주기

        elif isParked == 2 and distance < DISTANCE:  ### 불법차량일 때, 아직도 주차중이라면 신고
            if sendSMS == 0:
                # print("신고")
                account_sid = 'twilio sid'
                auth_token = 'twilio token'
                client = Client(account_sid, auth_token)

                message = client.messages \
                    .create(
                    body=exam + ' 차량이 불법 주차 하였습니다.',
                    from_='twilio phonenumber',
                    to='메시지 받을 phone number'
                )

                print(message.sid)

                sendSMS = 1
            else:
                continue

        elif distance > DISTANCE and isParked > 0:  ### 주차된 차가 있을 때, 차가 주차구역 벗어날 시
            isParked = 0  ### 주차된 차 없는 상태로 바꿔줌
            sendSMS = 0

#
#
except KeyboardInterrupt:  # Ctrl-C 입력 시
    GPIO.cleanup()  # GPIO 관련설정 Clear
    print("bye~")

# 번호판 입력 받아서 배열에 저장
# original_num=input()
# origin_list = list(original_num)


