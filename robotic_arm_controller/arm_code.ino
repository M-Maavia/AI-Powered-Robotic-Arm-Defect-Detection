/*

 * ============================================================

 *  AI-Powered Robotic Arm — Arduino PCA9685 Servo Controller

 *  Hardware : Arduino Uno/Mega + PCA9685 via I2C

 *  Servos   : 6-DOF  (3x MG996R + 3x SG90)

 *  Protocol : Serial (USB from Raspberry Pi)

 *             RX format : "90,45,30,120,10,5\n"

 *             TX        : "DONE\n" after move completes

 * ============================================================

 */



#include <Wire.h>

#include <Adafruit_PWMServoDriver.h>



// ── PCA9685 instance (default I2C address 0x40) ─────────────

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);



// ── Servo pulse limits (tune per servo type) ─────────────────

//    MG996R  : ~500 µs – 2400 µs

//    SG90    : ~500 µs – 2400 µs  (very similar)

#define SERVO_FREQ   50          // 50 Hz PWM

#define USEC_MIN    500

#define USEC_MAX   2400



// ── Number of DOF ─────────────────────────────────────────────

#define NUM_SERVOS   6



// ── Smooth-motion parameters ──────────────────────────────────

#define STEP_DEG     1           // degrees per increment

#define STEP_DELAY_MS 12         // ms between steps (prevents power surge)



// ── Current angles (start at 90° / neutral) ──────────────────

float currentAngle[NUM_SERVOS] = {90, 90, 90, 90, 90, 90};



// ── Angle constraints per channel ────────────────────────────

//    {min, max}  — adjust to avoid mechanical limits

const int angleMin[NUM_SERVOS] = {  0,   0,   0,   0,   0,   0};

const int angleMax[NUM_SERVOS] = {180, 180, 180, 180, 180, 90};



// ─────────────────────────────────────────────────────────────

//  convert degrees → PCA9685 tick count

// ─────────────────────────────────────────────────────────────

uint16_t angleToPulse(float deg) {

  float usec = map(deg, 0.0, 180.0, USEC_MIN, USEC_MAX);

  // PCA9685 runs at SERVO_FREQ Hz → period = 1 000 000 / SERVO_FREQ µs

  // 4096 ticks per period

  float ticksPerUsec = (4096.0 * SERVO_FREQ) / 1000000.0;

  return (uint16_t)(usec * ticksPerUsec);

}


void setServoImmediate(uint8_t ch, float deg) {

  deg = constrain(deg, angleMin[ch], angleMax[ch]);

  pwm.setPWM(ch, 0, angleToPulse(deg));

  currentAngle[ch] = deg;

}



// ─────────────────────────────────────────────────────────────

//  Move all 6 servos smoothly to target angles

//  Servos move in lock-step, one degree at a time.

// ─────────────────────────────────────────────────────────────

void moveAllSmooth(float target[NUM_SERVOS]) {

  // Clamp targets

  for (int i = 0; i < NUM_SERVOS; i++) {

    target[i] = constrain(target[i], angleMin[i], angleMax[i]);

  }



  bool moving = true;

  while (moving) {

    moving = false;

    for (int i = 0; i < NUM_SERVOS; i++) {

      float diff = target[i] - currentAngle[i];

      if (abs(diff) < 0.5) {

        // Close enough — snap to target

        if (currentAngle[i] != target[i]) {

          setServoImmediate(i, target[i]);

        }

        continue;

      }

      moving = true;

      float step = (diff > 0) ? STEP_DEG : -STEP_DEG;

      currentAngle[i] += step;

      pwm.setPWM(i, 0, angleToPulse(currentAngle[i]));

    }

    if (moving) delay(STEP_DELAY_MS);

  }

}



// ─────────────────────────────────────────────────────────────

//  Parse "a0,a1,a2,a3,a4,a5" from Serial into float array

//  Returns true on success.

// ─────────────────────────────────────────────────────────────

bool parseAngles(String line, float out[NUM_SERVOS]) {

  line.trim();

  int idx = 0;

  int start = 0;

  for (int c = 0; c <= line.length(); c++) {

    if (c == (int)line.length() || line[c] == ',') {

      if (idx >= NUM_SERVOS) return false;

      out[idx++] = line.substring(start, c).toFloat();

      start = c + 1;

    }

  }

  return (idx == NUM_SERVOS);

}



// ─────────────────────────────────────────────────────────────

//  setup()

// ─────────────────────────────────────────────────────────────

void setup() {

  Serial.begin(115200);

  Wire.begin();



  pwm.begin();

  pwm.setOscillatorFrequency(27000000);   // calibrate if needed

  pwm.setPWMFreq(SERVO_FREQ);

  delay(10);


  for (int i = 0; i < NUM_SERVOS; i++) {

    if (i == 5) {

      setServoImmediate(i, 5);

    } else {


      setServoImmediate(i, 90);

    }

    delay(50);  

  }

 

  Serial.println("READY");

}



// ─────────────────────────────────────────────────────────────

//  loop()

// ─────────────────────────────────────────────────────────────

void loop() {

  if (Serial.available()) {

    String line = Serial.readStringUntil('\n');

    line.trim();



    if (line.length() == 0) return;



    // ── Special commands ─────────────────────────────────────

    if (line.equalsIgnoreCase("PING")) {

      Serial.println("PONG");

      return;

    }



    if (line.equalsIgnoreCase("STATUS")) {

      Serial.print("ANGLES:");

      for (int i = 0; i < NUM_SERVOS; i++) {

        Serial.print((int)currentAngle[i]);

        if (i < NUM_SERVOS - 1) Serial.print(",");

      }

      Serial.println();

      return;

    }



    // ── Angle command ─────────────────────────────────────────

    float targets[NUM_SERVOS];

    if (parseAngles(line, targets)) {

      moveAllSmooth(targets);

      Serial.println("DONE");

    } else {

      Serial.println("ERROR:BAD_FORMAT");

    }

  }

}
