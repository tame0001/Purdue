import schedule
import redis
import json

redisdb = redis.Redis(host='localhost', port=6379, db=0)

def  tenHz():
	''' print current status and sensor value every 100 ms seconds '''
	print('---------------------------------------------')
	print("Encoder : ", json.loads(redisdb.get('encoder_value').decode('ascii')))    
	# print("Speed Read: ", json.loads(redisdb.get('speed').decode('ascii')))
	print("Front IR : ", redisdb.get('front_ir').decode('ascii'))
	print("Ball IR : ", redisdb.get('ball_ir').decode('ascii'))
	print("IR Line : ", redisdb.get('line_value').decode('ascii'))
	print("Left Line : ", redisdb.get('line_left').decode('ascii'))
	print("Speed Set : ", redisdb.get('driving_speed').decode('ascii'))
	print("Ball State : ", redisdb.get('ball_state').decode('ascii'))
	# print("Debug : ", redisdb.get('debug').decode('ascii'))
	

	line_value = redisdb.get('line_value').decode('ascii')
    
	result = 0
	try:
    
		result += int(line_value[0])*-4
		result += int(line_value[1])*-3
		result += int(line_value[2])*-2
		result += int(line_value[3])*-1
		result += int(line_value[4])*1
		result += int(line_value[5])*2
		result += int(line_value[6])*3
		result += int(line_value[7])*4

		if result == 0:
			zero_found = False
			for ir in line_value:
				if ir == '0':
					zero_found = True
            
			if not zero_found:
				result = 111
		
		print('IR Line : ', result)
		redisdb.set('line_result', result)
	
	except Exception as e:
		print(e)

def main():
	schedule.every(0.1).seconds.do(tenHz)
	while True:
		schedule.run_pending()


if __name__=='__main__':
	main()