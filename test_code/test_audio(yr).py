import pygame
import time

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500

pygame.mixer.init()

music_file = "C:/Users/yurim/Desktop/2020-1학기/캡스톤1/팀알라미(알람유)/capstone-alarmy/test_sound.mp3"

#SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#pygame.display.set_caption("pygame sound test")

clock = pygame.time.Clock()



pygame.mixer.music.load(music_file)
pygame.mixer.music.play()
time.sleep(10)

#pygame.mixer.quit()