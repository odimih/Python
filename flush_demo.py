import time
for num in range(10, 0, -1):
	print(str(num)+' ', end='', flush=True)
	time.sleep(1)
print('\nCountdown finished!')
