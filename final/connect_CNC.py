import serial
import time

# Initial Setup
# Set Serial port (포트를 정확히 확인하세요)
class Serial:
    def __init__(self, serial_port):
        self.ser = self.connect_serial(serial_port)

    def connect_serial(self, serial_port):
        '''
        Connect to the serial port and return it.
        '''
        ser = serial.Serial(serial_port, 9600, timeout=None)
        print("Communication Successfully started") # 연결 확인용
        return ser


    # 게임보드 그리는 명령을 보내는 함수
    def draw_game_board(self):
        '''
        Send 'S'(draw game board) command to CNC robot.
        '''
        while True:
            self.ser.write('S'.encode()) # 명령 전송
            print("Sent command: \'S\'")
            
            self.waiting_robot()
            break


    # 해당 신호를 보내는 함수
    def send_to_robot(self, command):
        '''
        Send 'command' to the Arduino one after the other.
        command[0] == 'X' or 'O'
        command[1] == position:int
        '''
        while True:
            time.sleep(1)
        
            self.ser.write(command.encode())  # 명령 전송
            print(f"Sent command: {command}")

            self.waiting_robot()
            break


    def waiting_robot(self):
        '''
        Waiting while receiving a serial signal from the Arduino and reading it.
        '''
        while True:
            if self.ser.in_waiting > 0:  # 수신된 데이터가 있으면
                response = self.ser.readline().decode()  # 데이터 읽기
                print(f"Arduino Response: {response}")
                break

# 코드 예시
# if __name__=="__main__":
#     ser = Serial('/dev/cu.usbserial-10')
#     ser.waiting_robot()

#     ser.draw_game_board()
#     ser.send_to_robot('X1')
    