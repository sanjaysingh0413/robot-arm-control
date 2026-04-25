# 🤖 3D Printed Affordable Robotic Arm

Build your own **low-cost industrial-style robotic arm** using 3D printed parts, Arduino control, and Python simulation tools. Designed for **students, makers, researchers, and small-scale automation projects**, this open-source project makes robotics affordable, customizable, and easy to learn.

---

## 🚀 Overview

This repository provides everything needed to build and control a functional robotic arm:

- ✅ 3D printable mechanical parts  
- ✅ Electronics wiring and circuit diagrams  
- ✅ Arduino firmware for motion control  
- ✅ Troubleshooting code  
- ✅ Python simulation notebooks  
- ✅ Full build guide and Bill of Materials  

Whether you're learning robotics, prototyping automation, or building your portfolio — this project is for you.

---

🛠 Features
💰 Affordable and beginner-friendly  
🖨 Fully 3D printable design  
⚙️ Arduino based control system  
🤖 5 DOF robotic arm  
🧠 Python simulation support  
🔌 Easy wiring and assembly  
📚 Great for education and projects  
🔧 Fully customizable and upgradeable  

---
🎯 Calibration (IMPORTANT)

Before first use:

⚠️ Disconnect servos from links  
⚠️ Allow all servos to rotate freely  
⚠️ Upload zero-position code first  

This prevents mechanical damage and ensures proper alignment.

---

🖨 3D Printing Settings

| Setting | Value |
|--------|------|
| Material | PLA / ABS |
| Layer Height | 0.2 mm |
| Infill | 20% |
| Base Infill | 50% |
| Supports | No |


---

⚡Electronics Setup

The robot uses standard servo motors connected to an Arduino controller.

Servo Layout (Top to Bottom)  
Gripper Servo (Hand)  
Wrist Servo  
Elbow Servo  
Shoulder Servo  
Base Servo  
Recommended Servos  
Joint	Servo Type  
Gripper	SG90  
Wrist	MG996R  
Elbow	MG996R  
Shoulder	High Torque Servo  
Base	MG996R  

---

Wiring Diagram
<div align="center"> <img width="600" src="https://github.com/user-attachments/assets/9e1a0557-dbdc-4448-9370-accc07058d7d"> </div>

---

**Main Components**

| Component | Quantity |
|-----------|----------|
| Arduino Board | 01 |
| MG996R Servo | As Required |
| SG90 Servo | 01 |
| Power Supply | 01 |
| Jumper Wires | Several |
| Screws / Nuts | As Required |
| PLA / ABS Filament | As Needed |

---

💻 **Software Installation**  
Required Tools  
Python 3.8+ → https://www.python.org  
Arduino IDE → https://www.arduino.cc/en/software  
Fusion 360 → https://www.autodesk.com/products/fusion-360  
Jupyter Notebook → https://docs.jupyter.org  
Pinocchio → https://stack-of-tasks.github.io/pinocchio/  
Meshcat → https://pypi.org/project/meshcat/  
Conda → https://anaconda.org/anaconda/conda  

---

**Running the code**  
Before running the code for the first time, make sure that all the servos are physically disconnected from the links and are allowed to move freely, for them to be able to move to the zero postion for calibration.
