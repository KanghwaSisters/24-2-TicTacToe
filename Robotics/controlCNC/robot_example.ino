#include "StepMotor.h"

Stepper stepper_x(8, 9, 10, 11);
Stepper stepper_y(3, 4, 5, 6);
Stepper stepper_z(A5, A4, A3, A2);

TicTacToeArtist artist(21, 21, &stepper_x, &stepper_y, &stepper_z);

void setup(){
    Serial.begin(115200);
    Serial.println("Setup complete");
};

void loop() {
    Serial.println("delay");
    delay(3000);
    // artist.move(1, 1, 1, 1);
    // artist.move(2, 15, -1, -1);
    // artist.move(3, 3, -1, -1);
    // delay(100000);
    Serial.println("start");
    artist.resetPosition();

    // artist.drawGameBoard();
    // artist.move(2, 2, -1, -1);
    // artist.move(5, 5,-1,-1);
    // artist.penUp();
    // artist.penDown();
    // delay(2000);
    // artist.penUp();
    // artist.move(1, 10, -1, -1);
    // artist.move(21, 21, 1, -1);
    // artist.move(1000, 0, -1,-1);
    // artist.move(2000, 2000, -1, -1);
    // artist.move(0, 1000, -1, 1);
    // artist.move(0, 1000, -1, -1);

    // Serial.println("Drawing game board...");
    // artist.drawGameBoard();   // 게임 보드 그리기
    // delay(2000);              // 잠시 대기

    // Serial.println("Drawing X...");
    // artist.drawX(0);           // 'X' 그리기
    // delay(2000);              // 잠시 대기
    
    Serial.println("Drawing Circle...");
    artist.drawCircle(0);      // 'O' 그리기
    delay(2000);              // 잠시 대기

    // Serial.println("Drawing X...");
    // artist.drawX(2);           // 'X' 그리기
    // delay(2000);              // 잠시 대기

    // Serial.println("Drawing Circle...");
    // artist.drawCircle(3);      // 'O' 그리기
    // delay(2000);              // 잠시 대기

    // Serial.println("Drawing X...");
    // artist.drawX(4);           // 'X' 그리기
    // delay(2000);              // 잠시 대기
    
    // Serial.println("Drawing Circle...");
    // artist.drawCircle(5);      // 'O' 그리기
    // delay(2000);              // 잠시 대기

    // Serial.println("Drawing X...");
    // artist.drawX(6);           // 'X' 그리기
    // delay(2000);              // 잠시 대기

    // Serial.println("Drawing Circle...");
    // artist.drawCircle(7);      // 'O' 그리기
    // delay(2000);              // 잠시 대기
    // // // Serial.println("start");
    // // // artist.move(1000, 0, -1, 1);

    // Serial.println("Drawing X...");
    // artist.drawX(8);           // 'X' 그리기
    // delay(2000);              // 잠시 대기

    Serial.println("end");
    delay(1000000);
};