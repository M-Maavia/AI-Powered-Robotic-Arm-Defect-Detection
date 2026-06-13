# AI-Powered Robotic Arm for Automated Defect Detection and Sorting of Bottled Products

An open-source, low-cost automation solution designed for Small and Medium Enterprises (SMEs) to perform real-time quality control inspection and automated sorting using Deep Learning and Edge Computing.

---

## ?? Key Features
* **6-DOF Serial-Chain Control:** Precise robotic manipulation for pick-and-place sorting based on defect classification.
* **Real-Time Edge AI:** Custom fine-tuned **YOLOv8-Nano** model optimized for edge devices.
* **6-Class Defect Inspection:** Simultaneously detects:
  1. Full (Correctly filled bottle)
  2. Underfilled
  3. Overfilled
  4. Cap Present/Absent
  5. Label Present/Absent
  6. Deformed (Structural damage)
* **Local Web Dashboard:** User-friendly localized Flask interface for live camera streaming, system monitoring, and analytical insights.

---

## ??? System Architecture & Hardware Stack
The entire system was developed with a cost-optimized budget of **PKR 28,140**, making industrial automation highly accessible.

* **Main Processor (Edge AI):** Raspberry Pi 4 (8GB RAM)
* **Microcontroller (Actuation):** Arduino Uno
* **Servo Driver:** Adafruit PCA9685 $I^2C$ PWM Driver
* **Actuators:** High-torque MG996R Servos (Base & Joints) + SG90 Servos (Gripper)
* **Vision System:** High-definition USB Camera module enclosed in a matte-black light tunnel with an LED ring light to eliminate specular reflections.

---

## ?? Performance & Results
* **AI Model Throughput:** **16.3 FPS** achieved on Raspberry Pi 4 using INT8 post-training quantization.
* **Model Accuracy:** Overall **0.826 mAP@0.5** (Validation dataset accuracy peaked at **0.944 mAP**).
* **Highest Performing Class:** Cap-off detection achieved **99.1%** accuracy.
* **End-to-End Latency:** Total system response time (from image capture to physical robotic sorting) is just **136 ms**.
* **Physical Manipulation Success Rate:** **93.3%** successful pick-and-place accuracy over extended testing cycles.

> ?? **Industrial Utility Note:** During live factory simulation testing, the system successfully flagged a mechanical drift in the upstream filling valve by identifying a continuous block of 260 overfilled bottles against only 25 correct ones, proving its capability to detect upstream machinery faults in real-time.

---

## ?? Repository Structure
```text
?? FYP_Project
?
??? ?? templates            # Flask HTML dashboard interfaces
??? ?? arduino_firmware     # Arduino (.ino) sketches for PCA9685 & servo joints
??? ?? hardware_design      # System schematics, circuit diagrams, and CAD models
?
??? ?? arm_controller.py    # Main integration script (YOLO Inference + Flask + Serial Control)
??? ?? best.pt              # Fine-tuned YOLOv8-Nano model weights
??? ?? requirements.txt     # Python dependencies
??? ?? README.md            # Project documentation (This file)