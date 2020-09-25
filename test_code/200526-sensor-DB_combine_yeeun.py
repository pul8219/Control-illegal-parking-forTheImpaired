import RPi.GPIO as GPIO  # RPi.GPIO에 정의된 기능을 GPIO라는 명칭으로 사용
import time
import picamera  # time 모듈

GPIO.setmode(GPIO.BCM)  # GPIO 이름은 BCM 명칭 사용
GPIO.setup(17, GPIO.OUT)  # Trig=17 초음파 신호 전송핀 번호 지정 및 출력지정
GPIO.setup(18, GPIO.IN)  # Echo=18 초음파 수신하는 수신 핀 번호 지정 및 입력지정
camera = picamera.PiCamera()
isParked = 0  ### 차 주차되어있는지 확인 0: 없을 때/ 1: 장애인차량 주차/ 2: 불법차량 주차
print("start")

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
        if distance < 10 && isParked == 0:  ### 주차된 차 없을 때, 차가 주차구역 진입시
            print("Take a picture")
            camera.capture('auto-picture-' + str(distance) + 'cm.jpg')
            isParked = checkDB()  ### 사진찍기
            time.sleep(30)  ### 차 빼는 시간 기다려주기

        elif isParked == 2 && distance < 10:  ### 불법차량일 때, 아직도 주차중이라면 신고
            #신고
        elif distance > 10 && isParked > 0:  ### 주차된 차가 있을 때, 차가 주차구역 벗어날 시
            isParked = 0  ### 주차된 차 없는 상태로 바꿔줌


except KeyboardInterrupt:  # Ctrl-C 입력 시
    GPIO.cleanup()  # GPIO 관련설정 Clear
    print("bye~")