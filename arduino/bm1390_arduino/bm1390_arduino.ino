/*****************************************************************************
   BM1390GLV.ino

   Copyright (c) 2021 ROHM Co.,Ltd.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 ******************************************************************************/
#include <Arduino.h>
#include <HardwareSerial.h>
#include <Wire.h>
#include <BM1390GLV.h>
#include <PID_v1.h>  // Include the PID library

#define SYSTEM_BAUDRATE (115200)
#define INO_VERSION     ("1.0")
#define SYSTEM_WAIT     (50)       // ms
#define ERROR_WAIT      (1000)      // ms
#define MAX_MOTOR_SPEED 250        // Maximum motor speed (PWM value)
#define REF_PRESSURE    1013.25    // Zero-Reference pressure in hPa 
#define PRESSURE_BIAS   6.508 // 3 Sync with py
// #define SS_PRESSURE     1021.25 // 1013.25 == 0 mmHg, 1021.25 == 6 mmHg, 1026.58 == 10 mmHg, 1029.24864 == 12mmHg, 1034.58152 == 16mmHg, 1041.24762 == 21mmHg, 1079.911 == 50 mmHg
#define SS_PRESSURE     1026.58 // + 0.8mmHg (bias)

#define AIR_PIN     10// White. PWM pin to control motor speed
#define In1         9 // Black. motor
#define In2         8 // Brown. motor
#define SUCTION_PIN 5 // Blue
#define In3         7 // Grey. solenoid valve
#define In4         6 // Purple solenoid valve
// #define PWMA        11
// #define DIRA        3

bool stopFlag = false;  // Sync with py A flag to indicate stop condition
bool motorControlEnabled = false;  // Sync with py Flag to enable/disable motor control
bool airControlEnabled = false; // Sync with py
bool suctionControlEnabled = false; // Sync with py

BM1390GLV bm1390glv;
// int motorSpeed1 = 0;
// int motorSpeed2 = 0;

double Setpoint = SS_PRESSURE;
double sensorReading, motorSpeed1=0.0, motorSpeed2=0.0, motorSpeed3=0.0;

// const int motorSpeeds[10] = {117, 132, 146, 160, 174, 190, 205, 224, 241, 250};
// const int stepDelay = 1 * 1000; // 610 seconds in milliseconds
// const int repeatCount = 5;

// PID tunings (you may need to adjust these)
double Kp1 = 75 , Ki1 = 75, Kd1 = 1;  // For air motor
// double Kp2 = 2.0, Ki2 = 5.0, Kd2 = 1.0;  // For suction motor
// double Kp2 = 13.0, Ki2 = 7.0, Kd2 = 1.0;  // For suction motor liitle faster but over shoot stays
// double Kp2 = 13.0, Ki2 = 3.0, Kd2 = 1.0;  // For suction motor liitle slow but near ss
// double Kp2 = 2.0, Ki2 = 3.0, Kd2 = 1.0;  // For suction motor starts too late but near ss
// double Kp2 = 2.0, Ki2 = 5.0, Kd2 = 1.0;  // For suction motor overshoot stay
// double Kp2 = 33.0, Ki2 = 23.0, Kd2 = 1.0;  // For suction motor sfsg
// double Kp2 = 33.0, Ki2 = 47.0, Kd2 = 1.0;  // For suction motor 
// double Kp2 = 16.03 , Ki2 = 16.32, Kd2 = 19.32; // Initial PID parameters, best stab time: 4011 seconds
double Kp2 = 75, Ki2 = 75, Kd2 = 1;  // For suction motor 

// Initialize PID controllers for air and suction motors
PID airPID(&sensorReading, &motorSpeed1, &Setpoint, Kp1, Ki1, Kd1, DIRECT); // REVERSE for more far, more faster
PID suctionPID(&sensorReading, &motorSpeed2, &Setpoint, Kp2, Ki2, Kd2, REVERSE);

// // For test PWM
// int pwmTEST = 0;

void error_func(int32_t result);
// Helper function to stop the motors
void stopAirMotors();
void stopSuctionMotors();

void setup()
{
    int32_t result;

   // Initialize motor control pins
    pinMode(AIR_PIN, OUTPUT);
    pinMode(In1, OUTPUT);
    pinMode(In2, OUTPUT);
    pinMode(SUCTION_PIN, OUTPUT);
    pinMode(In3, OUTPUT);
    pinMode(In4, OUTPUT);
    // pinMode(PWMA, OUTPUT);
    // pinMode(DIRA, OUTPUT);
    
    // digitalWrite(DIRA, HIGH);
    Serial.begin(SYSTEM_BAUDRATE);
    while (!Serial) {
    }

    Wire.begin();

    result = bm1390glv.init();
    if (result == BM1390GLV_COMM_OK) {
        (void)bm1390glv.start();
    } else {
        error_func(result);
    }

    // Initialize PID controllers
    airPID.SetMode(AUTOMATIC);
    airPID.SetOutputLimits(0, MAX_MOTOR_SPEED);

    suctionPID.SetMode(AUTOMATIC);
    suctionPID.SetOutputLimits(0, MAX_MOTOR_SPEED);

    return;
}

void loop()
{
    int32_t result;
    float32 press, temp;
    
    // analogWrite(PWMA, 200);
    // digitalWrite(DIRA, LOW);

    // Check for quitting
    if (stopFlag) {
        // Stop the loop execution
        // Serial.println("Stopping program...");
        // turn off motor and solenoid valve
        stopAirMotors();
        stopSuctionMotors();

        // digitalWrite(DIRA, LOW);
        // analogWrite(PWMA, 0);

        Serial.flush();  // Wait for all outgoing serial data to be sent
        Serial.end();    // Optionally "end" the serial communication
        while(true);  // Halt the program here
    }

    // // Motor PWM control logic
    // // if (motorControlEnabled) {
    //     for (int repeat = 0; repeat < repeatCount; repeat++) {
    //         for (int i = 1; i < 16; i++) {
    //             delay(5000);  // Reduced delay for faster updates
    //             // int pwmValue = motorSpeeds[i];
    //             Serial.print("Setting MotorA PWM to: ");
    //             Serial.println(i*10+100);
    //         }
    //     }
    //     // motorControlEnabled = false;  // Disable motor control after the sequence
    // // }

    result = bm1390glv.get_val(&press, &temp);
    if (result == BM1390GLV_COMM_OK) {
        sensorReading = press + PRESSURE_BIAS; // Update the PID input

        (void)Serial.print(press);
        // (void)Serial.write(" ");
        // (void)Serial.print(pwmTEST);
        (void)Serial.print(" ");
        (void)Serial.print(temp);
        (void)Serial.print(" ");
        (void)Serial.print(motorSpeed1);
        (void)Serial.print(" ");
        (void)Serial.println(motorSpeed2);

        if (airControlEnabled && motorControlEnabled) {
            analogWrite(AIR_PIN, motorSpeed1); // Control motor speed based on received value
            digitalWrite(In1, HIGH);
            digitalWrite(In2, LOW);
            if (sensorReading > SS_PRESSURE) { // 0.3545 hPa == 0.2659 mmHg from Standard error
                suctionPID.Compute();
                analogWrite(SUCTION_PIN, motorSpeed2); 
                digitalWrite(In3, HIGH);
                digitalWrite(In4, LOW);
            } else {
                stopSuctionMotors();
            }
        } else if (suctionControlEnabled && motorControlEnabled) {
            analogWrite(SUCTION_PIN, motorSpeed2); // Control motor speed based on received value
            digitalWrite(In3, HIGH);
            digitalWrite(In4, LOW);
            if (sensorReading < SS_PRESSURE) { // 0.3545 hPa == 0.2659 mmHg from Standard error
                airPID.Compute();
                analogWrite(AIR_PIN, motorSpeed1);
                digitalWrite(In1, HIGH);
                digitalWrite(In2, LOW);
            } else {
                stopAirMotors();
            }
        } else if (airControlEnabled && !motorControlEnabled) {
            analogWrite(AIR_PIN, motorSpeed1); // Control motor speed based on received value
            digitalWrite(In1, HIGH);
            digitalWrite(In2, LOW);
            stopSuctionMotors();
        } else if (suctionControlEnabled && !motorControlEnabled) {
            analogWrite(SUCTION_PIN, motorSpeed2); // Control motor speed based on received value
            digitalWrite(In3, HIGH);
            digitalWrite(In4, LOW);
            stopAirMotors();
        } else if (motorControlEnabled) {
            if (sensorReading > SS_PRESSURE) { // 0.3545 hPa == 0.2659 mmHg from Standard error
                suctionPID.Compute();
                analogWrite(SUCTION_PIN, motorSpeed2); 
                digitalWrite(In3, HIGH);
                digitalWrite(In4, LOW);
                stopAirMotors();
            }
            if (sensorReading < SS_PRESSURE) { // 0.3545 hPa == 0.2659 mmHg from Standard error
                airPID.Compute();
                analogWrite(AIR_PIN, motorSpeed1); 
                digitalWrite(In1, HIGH);
                digitalWrite(In2, LOW);
                stopSuctionMotors();
            }
                // stopAirMotors();
                // stopSuctionMotors();
            // }
        } else {
            stopAirMotors();
            stopSuctionMotors();
        }
            // if (motorControlEnabled) {
            //     // if (millis() >= 0) {
            //     //     motorSpeed1 = abs(255 * sin ( millis()/20000.0 * PI));
            //     // }
            //     analogWrite(AIR_PIN, motorSpeed1); // Control motor speed based on received value
            //     if (motorSpeed1 > 0) {
            //         digitalWrite(In1, HIGH);
            //         digitalWrite(In2, LOW);
            //     } else {
            //         digitalWrite(In1, LOW);
            //         digitalWrite(In2, LOW);
            //     }
            // } else {
            //     // Turn off motor and solenoid valve if motor control is disabled
            //     analogWrite(AIR_PIN, 0);
            //     digitalWrite(In1, LOW);
            //     digitalWrite(In2, LOW);
            // }
    } else {
        error_func(result);
    }

    // // motor pwm test
    // if (millis() >= 0) {
    //     pwmTEST = abs(255 * sin ( millis()/5000.0 * PI));
    // }
    // analogWrite(AIR_PIN, pwmTEST);
    // pwmTEST >= 200 ? digitalWrite(SUCTION_PIN, pwmTEST) : digitalWrite(SUCTION_PIN, LOW);

    // Check for incoming serial data for quitting or motor control
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');  // Read the input until a newline character

        input.trim();  // Remove any leading or trailing whitespace
        // String input = Serial.readString();
        if (input.indexOf("Q") >= 0) { // If message is 'Q', stop serial comm.
            stopFlag = true;  // Set the flag to stop the loop
        } else if (input.indexOf("MOTOR_ON") >= 0) {
            motorControlEnabled = true;
            // Serial.println("Motor control enabled.");
        } else if (input.indexOf("MOTOR_OFF") >= 0) {
            motorControlEnabled = false;
            // Serial.println("Motor control disabled.");
        } else if (input.indexOf("AIR_ON") >= 0) {
            airControlEnabled = true;
            // Serial.println("Air manual control enabled.");
        } else if (input.indexOf("AIR_OFF") >= 0) {
            motorSpeed1 = 0;
            airControlEnabled = false;
            // Serial.println("Air manual control disabled.");
        } else if (input.indexOf("SUCTION_ON") >= 0) {
            suctionControlEnabled = true;
            // Serial.println("Suction manual control enabled.");
        } else if (input.indexOf("SUCTION_OFF") >= 0) {
            motorSpeed2 = 0;
            suctionControlEnabled = false;
            // Serial.println("Suction manual control disabled.");
        } else if (input.startsWith("A_SPEED ")) {
            motorSpeed1 = constrain(input.substring(8).toInt(), 0, MAX_MOTOR_SPEED);
            // Serial.print("Air speed set to: ");
            // Serial.println(motorSpeed1);
        } else if (input.startsWith("S_SPEED ")) {
            motorSpeed2 = constrain(input.substring(8).toInt(), 0, MAX_MOTOR_SPEED);
            // Serial.print("Suction speed set to: ");
            // Serial.println(motorSpeed2);
            // int newSpeed = input.substring(6).toInt();
            // motorSpeed1 = constrain(newSpeed, 0, MAX_MOTOR_SPEED);
            // Serial.print("Motor speed set to: ");
            // Serial.println(motorSpeed1);
        // } else if (input.startsWith("AUTO_ENV_ON")) {
        //     motorSpeed3 = constrain(input.substring(8).toInt(), 0, MAX_MOTOR_SPEED);
        //     // Serial.print("Suction speed set to: ");
        //     // Serial.println(motorSpeed2);
        //     // int newSpeed = input.substring(6).toInt();
        //     // motorSpeed1 = constrain(newSpeed, 0, MAX_MOTOR_SPEED);
        //     // Serial.print("Motor speed set to: ");
        //     // Serial.println(motorSpeed1);
        } else {
            // Serial.println("Unknown command");
        }
    }
  
    delay(SYSTEM_WAIT);  // Reduced delay for faster updates
}

// Helper function to stop the motors
void stopAirMotors() {
    analogWrite(AIR_PIN, 0);
    digitalWrite(In1, LOW);
    digitalWrite(In2, LOW);
}

void stopSuctionMotors() {
    analogWrite(SUCTION_PIN, 0);
    digitalWrite(In3, LOW);
    digitalWrite(In4, LOW);
}
// void MotorA(int pwm, boolean reverse) {
//   analogWrite(PWMA, pwm);
//   if (reverse) {
//     digitalWrite(DIRA, HIGH);
//   }
//   else {
//     digitalWrite(DIRA, LOW);
//   }
// }

void error_func(int32_t result)
{
    uint8_t cnt;

    switch (result) {
        case BM1390GLV_COMM_ERROR:
            (void)Serial.println("Communication Error.");
            break;

        case BM1390GLV_WAI_ERROR:
            (void)Serial.println("ID Error.");
            break;

        default:
            (void)Serial.println("Unknown Error.");
            break;
    }

    (void)Serial.println("BM1390GLV Check System and Driver Parameter.");
    cnt = 0;
    while (1) {
        (void)Serial.print(".");
        if (cnt < 30) {
            cnt++;
        } else {
            cnt = 0;
            (void)Serial.println();
        }
        delay(ERROR_WAIT);
    }
}
