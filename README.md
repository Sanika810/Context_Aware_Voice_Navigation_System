# Context_Aware_Voice_Navigation_System
Enhanced Context-Aware Voice 
Navigation System 
Using YOLO-Based Object Detection and Neuro-Fuzzy 
Logic 
Course Title: Neuro-Fuzzy Techniques 
Team Members: 
● Mrummayee Limaye (BT23CSE019) 
● Anagha Choudhary (BT23CSE034) 
● Sanika Budhe (BT23CSE036) 


1. Introduction 
The Context-Aware Voice Navigation System is an intelligent driving assistance system that 
enhances road safety through real-time object detection and intelligent decision-making. By 
integrating YOLOv8 for object detection and a neuro-fuzzy inference system for contextual 
analysis, the system provides voice-guided navigation instructions, obstacle avoidance alerts, 
and turn recommendations. 
Key Features: 
● Real-time object detection using YOLOv8 
● Fuzzy logic-based decision making 
● Voice-guided navigation instructions 
● Multi-zone spatial analysis 
● Emergency evasion suggestions 
● Video processing capability


3. System Requirements 
2.1 Software Requirements 
# Core Dependencies 
pip install ultralytics opencv-python scikit-fuzzy numpy 
# Additional Utilities 
pip install pathlib json math collections time threading datetime 


2.2 Hardware Requirements 
● Webcam (for real-time mode) 
● 4GB+ RAM 
● Python 3.7+ 


2.3 Platform Compatibility 
● Recommended: Ubuntu/Linux (optimal performance) 
● Supported: Windows, macOS 
● Note: Due to multithreading implementation for voice integration, the program works 
optimally on Ubuntu systems with better thread management and audio handling 
capabilities. 


3. Installation & Setup 
3.1 File Structure 
project/ 
├── main.py              
├── speak.py             
├── dataset/            
# Main program file 
# Voice synthesis module 
# Folder containing test images 
│   └── images/         
└── results/            
# Place your test images here 
# Output directory (auto-created) 


3.2 Running the System 
python main.py 


4. Available Operation Modes 
4.1 Real-time Camera Feed (Option 1) 
● Purpose: Live navigation assistance using webcam 
● Features: 
○ Real-time object detection and tracking 
○ Instant voice instructions 
○ Visual feedback with colored bounding boxes 
○ Spatial zone analysis 
● Controls: 
○ q - Quit application 
○ s - Save current frame 
○ p - Print current instruction to console


4.2 Process Dataset (Option 2) 
● Purpose: Batch processing of images from dataset folder 
● Features: 
○ Processes all images in dataset/images/ 
○ Generates comprehensive analysis reports 
○ Saves annotated images in results/ folder 
○ Creates statistical summary 


5. Technical Architecture 
5.1 Core Components 
5.1.1 YOLOv8 Object Detection 
● Pre-trained on COCO dataset (80 classes) 
● Real-time processing at 25-30 FPS 
● Custom filtered for navigation-relevant objects


5.1.2 Fuzzy Logic System 
Input Variables: 
● Distance (0-200 meters) 
● Confidence (0-1 detection certainty) 
● Priority (0-10 object importance) 
● Lateral Position (-90° to 90°) 
● Speed (0-120 km/h) 
Output Variables: 
● Urgency (0-100 criticality score) 
● Safety Score (0-100 safety assessment) 


5.1.3 Spatial Zone Analysis 
Five defined navigation zones: 
● Center (±15°): Direct path 
● Left (-45° to -15°): Left turn zone 
● Right (15° to 45°): Right turn zone 
● Far Left (-90° to -45°): Sharp left 
● Far Right (45° to 90°): Sharp right 


5.2 Voice Integration & Multithreading 
● Multithreading Architecture: Voice synthesis runs in separate threads to prevent 
blocking main processing pipeline 
● Ubuntu Optimization: Superior thread scheduling and audio subsystem in Linux 
provides seamless voice integration 
● Non-blocking Design: Object detection continues uninterrupted while voice instructions 
are generated 


5.3 Navigation Object Priorities 
| Object Type     | Priority | Stop Distance | Caution Distance |
|-----------------|----------|---------------|------------------|
| Person          | 10       | 5 m           | 35 m             |
| Traffic Light   | 10       | 15 m          | 50 m             |
| Stop Sign       | 10       | 10 m          | 30 m             |
| Car / Truck     | 8        | 8–10 m        | 25–30 m          |
| Bicycle         | 7        | 6 m           | 20 m             |


6. Output Examples 
6.1 Voice Instructions 
● Clear Path: "Path clear. Continue straight ahead." 
● Emergency Stop: "EMERGENCY STOP! Person at 5 meters. Left turn available for 
evasion." 
● Caution: "Vehicle ahead at 25 meters. Reduce speed and maintain safe distance." 
● Turn Opportunities: "Path clear. Left turn available with 50 meter clearance."


6.2 Visual Indicators 
● Green Boxes: Safe objects (>30m) 
● Yellow Boxes: Moderate urgency (15-30m) 
● Orange Boxes: High urgency (8-15m) 
● Red Boxes: Critical urgency (<8m) 


7. Performance Metrics 
● Processing Speed: 25-30 FPS (real-time) 
● Detection Accuracy: 85%+ on relevant objects 
● Voice Latency: <500ms response time 
● Memory Usage: ~1.5GB during operation 
● Platform Performance: Optimal on Ubuntu due to efficient multithreading


8. Platform-Specific Notes 
8.1 Ubuntu/Linux (Recommended) 
● Voice Integration: Seamless multithreading with no audio lag 
● Performance: Optimal frame rates and real-time processing


8.2 Windows 
● Voice Integration: Functional but may experience minor delays 
● Performance: Slightly reduced frame rates due to thread overhead 


9. Conclusion 
The Enhanced Context-Aware Voice Navigation System successfully demonstrates the 
integration of computer vision and fuzzy logic for intelligent navigation assistance. The system 
provides: 
✅
Real-time object detection and tracking 
✅
Intelligent contextual decision making 
✅
Clear voice-guided instructions 
✅
Multiple operation modes for flexibility 
✅
Comprehensive visual and statistical outputs 
✅
Optimized multithreading for seamless voice integration (especially on Ubuntu)
