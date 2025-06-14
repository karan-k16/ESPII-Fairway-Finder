# Fairway Finder: Markerless Golf Ball Position Encoder

This project presents the **Fairway Finder**, a computer vision-based system designed to determine the final x-y position of a golf ball without using any physical markers. Developed as part of a UofT design project under Professor Luc Tremblay, the system uses a ceiling-mounted camera and computer vision techniques to automate the tracking process in a lab putting green environment.

> 🛠️ **All code and image processing implementation in this repository was written solely by me, Karan Kardam.**

## 📌 Project Overview

The system processes an image of a golf ball on a putting green, removes camera distortion, detects the ball’s position, and converts it into real-world coordinates using homography.

## 🧠 Core Features

- **Markerless Detection** via HSV filtering and HoughCircles
- **Perspective Mapping** using a custom homography matrix
- **Camera Calibration** (optional) for real-world undistortion
- **GUI Output** showing location and measurement feedback
- **Sub-centimeter Accuracy** suitable for motor control studies

## 📂 Repository Structure

├── fairway_finder.py # Full implementation of the ball detection system 
├── images/ # (optional) calibration and test images
├── output/ # (optional) JSON outputs from detection
├── README.md # This file


## 🛠️ Tech Stack

- Python
- OpenCV
- NumPy
- Tkinter (for GUI output)
- JSON
- Blender (for synthetic image testing)

## ⚙️ How It Works

1. **Image Input**: Capture from overhead camera (or Blender simulation).
2. **Undistortion**: Optional correction using chessboard calibration images.
3. **Detection**: Circle detection after HSV-based masking isolates ball color.
4. **Homography**: Translates pixel coordinates into real-world meters.
5. **Output**: Shows image with annotations and saves displacement data in JSON.

## 🧪 Testing & Accuracy

- Blender-generated testing images were used to simulate lab conditions.
- The system maintains sub-centimeter accuracy (±5 mm) in all trials.
- Output includes displacement from reference in meters and visual feedback.

## 📄 Citation & Acknowledgment

This work was completed as part of **APS112: Engineering Strategies & Practice II** at the University of Toronto.

**Client:** Prof. Luc Tremblay  
**Team 140 Members:**  
Yuma Iwamoto, Hashim Sawan, Karan Kardam, Mouj Nagro, Jiayi Zhang, Chika Kameda

> 💻 **Note:** While the design report was submitted collaboratively, the full Python implementation (`fairway_finder.py`) and all vision algorithms were developed independently by **Karan Kardam**.

## 📬 Contact

**Karan Kardam**  
📧 karan.kardam@mail.utoronto.ca

