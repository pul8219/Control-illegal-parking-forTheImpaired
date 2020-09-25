import pymysql
import pygame
import time

#from pydub import AudioSegment
#from pydub.playback import play

conn = pymysql.connect(host='localhost', user='root', password='DB_passwd', db='alarmy', charset='utf8')
test_number = "67마1234"
#comp_number = ""
music_file = "C:/Users/yurim/Desktop/2020-1학기/캡스톤1/팀알라미(알람유)/capstone-alarmy/Control-illegal-parking-forTheHandicapped/test_sound.mp3"


try:

    pygame.mixer.init()
    pygame.mixer.music.load(music_file)

    cursor = conn.cursor();

    sql = "SELECT * FROM disabled_vehicle WHERE vehicle_number='" + test_number + "'"
    cursor.execute(sql)
    result_cnt = cursor.rowcount
    print(result_cnt)

    # 일치하는 결과가 없을 경우
    if result_cnt == 0:
        pygame.mixer.music.play()
        time.sleep(10)

    #for row in cursor.fetchall():
    #    print(row[0], row[1], row[2], row[3])

    #data = cursor.fetchall()
    #for row in data:
    #    comp_number = row[1]

    #print(comp_number)

    #pygame.mixer.init()
    #pygame.mixer.music.load(music_file)


    #song = AudioSegment.from_mp3("C:/Users/yurim/Desktop/2020-1학기/캡스톤1/팀알라미(알람유)/capstone-alarmy/test_sound.mp3")

    #if test_number != comp_number:
    #    pygame.mixer.music.play()
    #    time.sleep(10)


    #pygame.mixer.quit()

    conn.commit()


finally:
    conn.close()