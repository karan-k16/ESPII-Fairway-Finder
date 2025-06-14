# Fairway Finder: Markerless Golf Ball Position Encoder

This project presents the **Fairway Finder**, a computer vision-based system designed to capture the endpoint location of a golf ball without requiring physical markers. It was developed in collaboration with Professor Luc Tremblay at the University of Toronto as part of a study on motor skill acquisition in golf putting.

> ⚠️ **Note:** While the design and report were completed as part of a team assignment, **all code and technical implementation in this repository was developed independently by me**.

## 📌 Project Overview

The system outputs accurate x-y coordinates of a golf ball on an artificial green using an overhead camera. It leverages OpenCV for detection and homography transformations to convert image coordinates to real-world measurements.

## 🧠 Core Features

- **Markerless Detection** using image processing
- **Homography-Based Scaling** to map coordinates from image space to real-world dimensions
- **JSON Output** for seamless integration with analysis tools
- **Accuracy within ±5 mm**, suitable for lab-based motor control experiments

## 📂 Repository Structure

├── UTMIST-Image-Enhancement.ipynb # Code prototype for markerless golf ball detection
├── Tut26_Team140_CDS.docx.pdf # Full conceptual design report (team submission)
├── assets/ # Input images, checkerboard calibration (optional)
├── output/ # JSON output of ball positions (optional)
├── README.md # This file

## 🛠️ Tech Stack

- Python
- OpenCV
- NumPy
- JSON
- Blender (for 3D testing environment)

## ⚙️ How It Works

1. Capture a high-res image of the green after a putt.
2. Undistort the image using camera calibration techniques.
3. Detect the golf ball using image processing (thresholding, contour detection).
4. Apply a homography transformation to convert the image coordinates to real-world x-y positions.
5. Export the result to a `.json` file.

## 🧪 Testing & Accuracy

- Testing was conducted using synthetic 3D environments in Blender.
- 30 trials were performed and error was computed against known golf ball positions.
- Average accuracy met the ±5 mm requirement.

## 📄 Citation & Acknowledgments

This project was developed for **APS112: Engineering Strategies & Practice II** at the University of Toronto.

**Client:** Prof. Luc Tremblay  
**Design Report Team:**  
Yuma Iwamoto, Hashim Sawan, Karan Kardam, Mouj Nagro, Jiayi Zhang, Chika Kameda

> 🔧 **All code in this repository was written solely by Karan Kardam.**

## 📬 Contact

For any questions, collaborations, or clarifications:

**Karan Kardam**  
📧 karan.kardam@mail.utoronto.ca
