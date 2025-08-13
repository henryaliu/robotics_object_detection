import cv2
import sys
import numpy as np

# Preferred UI font for on-screen text
DEFAULT_FONT = cv2.FONT_HERSHEY_DUPLEX

"""
Multi-Color Shape Detection with Dictionary-Based Labeling

This program captures video from your webcam and performs simultaneous detection of multiple 
colors and shapes, labelling the detections based on word entries.

Features:
- Simultaneous detection of multiple colors (red, blue, green, yellow, orange, purple)
- Shape classification (triangle, rectangle, circle, pentagon, hexagon, etc.)
- Dictionary-based labeling system for color-shape combinations
- Real-time object identification with color-coded bounding boxes

Controls:
- Press 'q' to quit
- Toggle detection for specific colors using number keys (1-6)

HSV Color Ranges (different ranges of H, S, and V):
- 1. Red: H(0-10, 160-179), S(50-255), V(50-255)
- 2. Blue: H(100-130), S(50-255), V(50-255) 
- 3. Green: H(40-80), S(50-255), V(50-255)
- 4. Yellow: H(20-30), S(50-255), V(50-255)
- 5. Orange: H(10-20), S(50-255), V(50-255)
- 6. Purple: H(130-160), S(50-255), V(50-255)

Use the sliders to adjust these ranges.

"""

# Select the best backend, appropriate for the OS
def get_optimal_backend():
    preferred = []
    if sys.platform.startswith('win'):
        # Windows: 1. Media Foundation 2. DirectShow 
        if hasattr(cv2, 'CAP_MSMF'):
            preferred.append(cv2.CAP_MSMF)
        if hasattr(cv2, 'CAP_DSHOW'):
            preferred.append(cv2.CAP_DSHOW)
        preferred.append(cv2.CAP_ANY)
    elif sys.platform.startswith('linux'):
        # Linux: V4L2
        if hasattr(cv2, 'CAP_V4L2'):
            preferred.append(cv2.CAP_V4L2)
        preferred.append(cv2.CAP_ANY)
    else:
        preferred.append(cv2.CAP_ANY)
    return preferred


def backend_name(backend_value: int) -> str:
    name_map = {
        getattr(cv2, 'CAP_MSMF', -1): 'MSMF (Media Foundation)',
        getattr(cv2, 'CAP_DSHOW', -2): 'DSHOW (DirectShow)',
        getattr(cv2, 'CAP_V4L2', -3): 'V4L2',
        getattr(cv2, 'CAP_ANY', 0): 'ANY'
    }
    return name_map.get(backend_value, str(backend_value))


backends = get_optimal_backend()
camera_indices = list(range(0, 6))  # Try a range of indices in case 0 is absent.

cap = None
for backend in backends:
    for i in camera_indices:
        print(f"Trying camera {i} with backend {backend_name(backend)} ({backend})")
        cap = cv2.VideoCapture(i, backend)
        if cap is not None and cap.isOpened():
            print(f"Success! Using camera {i} with backend {backend_name(backend)}")
            break
        if cap is not None:
            cap.release()
            cap = None
    if cap is not None and cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera")
    print("WSL may not have access to Windows webcam. Try running in Windows Command Prompt instead.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

# Dictionary of colors defined by HSV values
color_ranges = {
    'red': {
        'lower1': np.array([0, 50, 50]),     # Red lower range 1
        'upper1': np.array([10, 255, 255]),   # Red upper range 1
        'bgr_color': (0, 0, 255),            # Red in BGR for drawing
        'enabled': False
    },
    'blue': {
        'lower1': np.array([100, 50, 50]),
        'upper1': np.array([130, 255, 255]),
        'bgr_color': (255, 0, 0),            # Blue in BGR
        'enabled': True
    },
    'green': {
        'lower1': np.array([40, 50, 50]),
        'upper1': np.array([80, 255, 255]),
        'bgr_color': (0, 255, 0),            # Green in BGR
        'enabled': False
    },
    'yellow': {
        'lower1': np.array([20, 50, 50]),
        'upper1': np.array([30, 255, 255]),
        'bgr_color': (0, 255, 255),          # Yellow in BGR
        'enabled': False
    },
    'orange': {
        'lower1': np.array([10, 50, 50]),
        'upper1': np.array([20, 255, 255]),
        'bgr_color': (0, 165, 255),          # Orange in BGR
        'enabled': False
    },
    'purple': {
        'lower1': np.array([130, 50, 50]),
        'upper1': np.array([160, 255, 255]),
        'bgr_color': (255, 0, 255),          # Purple in BGR
        'enabled': False
    }
}

# Dictionary for shape-color combinations and their labels
object_labels = {
    # Color + Shape combinations for intelligent labeling
    ('red', 'triangle'): 'Red Triangle',
    ('red', 'rectangle'): 'Red Rectangle',
    ('red', 'circle'): 'Red Circle',
    ('blue', 'triangle'): 'Blue Triangle',
    ('blue', 'rectangle'): 'Blue Rectangle',
    ('blue', 'circle'): 'Blue Circle',
    ('green', 'triangle'): 'Green Triangle',
    ('green', 'rectangle'): 'Green Rectangle',
    ('green', 'circle'): 'Green Circle',
    ('yellow', 'triangle'): 'Yellow Triangle',
    ('yellow', 'rectangle'): 'Yellow Rectangle',
    ('yellow', 'circle'): 'Yellow Circle',
    ('orange', 'triangle'): 'Orange Triangle',
    ('orange', 'rectangle'): 'Orange Rectangle',
    ('orange', 'circle'): 'Orange Circle',
    ('purple', 'triangle'): 'Purple Triangle',
    ('purple', 'rectangle'): 'Purple Rectangle',
    ('purple', 'circle'): 'Purple Circle'
}

# HSV tuning default color
current_tuning_color = 'blue'  # Default color
tuning_enabled = True

# Create trackbar window for HSV tuning
cv2.namedWindow('HSV Tuning Controls')

def update_color_range(color_name, h_min, s_min, v_min, h_max, s_max, v_max):
    """Update the HSV range for a specific color"""
    if color_name in color_ranges:
        color_ranges[color_name]['lower1'] = np.array([h_min, s_min, v_min])
        color_ranges[color_name]['upper1'] = np.array([h_max, s_max, v_max])

def load_color_to_trackbars(color_name):
    """Load a color's HSV values to the trackbars"""
    if color_name in color_ranges:
        color_info = color_ranges[color_name]
        lower = color_info['lower1']
        upper = color_info['upper1']
        
        cv2.setTrackbarPos('H Min', 'HSV Tuning Controls', lower[0])
        cv2.setTrackbarPos('S Min', 'HSV Tuning Controls', lower[1])
        cv2.setTrackbarPos('V Min', 'HSV Tuning Controls', lower[2])
        cv2.setTrackbarPos('H Max', 'HSV Tuning Controls', upper[0])
        cv2.setTrackbarPos('S Max', 'HSV Tuning Controls', upper[1])
        cv2.setTrackbarPos('V Max', 'HSV Tuning Controls', upper[2])

# Initialize trackbars with blue color values
initial_color = color_ranges[current_tuning_color]
h_min, s_min, v_min = initial_color['lower1']
h_max, s_max, v_max = initial_color['upper1']

# Create trackbars for HSV values, using OpenCV's given ranges.
cv2.createTrackbar('H Min', 'HSV Tuning Controls', int(h_min), 179, lambda x: None)
cv2.createTrackbar('S Min', 'HSV Tuning Controls', int(s_min), 255, lambda x: None)
cv2.createTrackbar('V Min', 'HSV Tuning Controls', int(v_min), 255, lambda x: None)
cv2.createTrackbar('H Max', 'HSV Tuning Controls', int(h_max), 179, lambda x: None)
cv2.createTrackbar('S Max', 'HSV Tuning Controls', int(s_max), 255, lambda x: None)
cv2.createTrackbar('V Max', 'HSV Tuning Controls', int(v_max), 255, lambda x: None)

# 'Press q to quit' text below trackbars
quit_box = np.zeros((50, 500, 3), dtype=np.uint8)
cv2.putText(quit_box, 'Press Q to Quit', (10, 30), 
            DEFAULT_FONT, 0.5, (255, 255, 255), 1)

print("Multi-Color Detection System started!")
print("Detecting: Red, Blue, Green, Yellow, Orange, Purple")
print("\n=== CONTROLS ===")
print("Number keys (1-6): Toggle detection for specific colors")
print("  1-Red, 2-Blue, 3-Green, 4-Yellow, 5-Orange, 6-Purple")
print("Letter keys (R,B,G,Y,O,P): Select color for HSV tuning")
print("HSV Tuning Controls: Use sliders to adjust ranges")
print("Q: Quit application")

def get_shape_name(vertices, area):
    """Classify shape based on number of vertices and additional characteristics"""
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        return "rectangle"
    elif vertices > 8:  # Many vertices usually indicate a circle
        return "circle"
    elif vertices == 5:
        return "pentagon"
    elif vertices == 6:
        return "hexagon"
    else:
        return f"{vertices}-sided polygon"

def detect_objects_by_color(hsv_frame, original_frame, color_name, color_info):
    """Detect objects of a specific color and return detection results"""
    detected_objects = []
    
    # Create color mask for this color (single HSV range)
    mask = cv2.inRange(hsv_frame, color_info['lower1'], color_info['upper1'])
    
    # Find contours, ignore hierarchy, keep outermost only (RETR_EXTERNAL), chain_approx_simple for fast & simple contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small contours
            # Get contour properties
            perimeter = cv2.arcLength(contour, True) # contour perimeter
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True) # simplify into polygon
            
            # Get bounding rectangle and center
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Classify shape
                vertices = len(approx)
                shape_name = get_shape_name(vertices, area)
                
                # Get intelligent label
                label_key = (color_name, shape_name)
                intelligent_label = object_labels.get(label_key, f"{color_name.title()} {shape_name.title()}")
                
                detected_objects.append({
                    'contour': contour,
                    'approx': approx,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'area': area,
                    'vertices': vertices,
                    'shape': shape_name,
                    'color': color_name,
                    'label': intelligent_label,
                    'draw_color': color_info['bgr_color']
                })
    
    return detected_objects, mask

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Handle keyboard input for toggling colors and tuning
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Toggle detection for colors (1-6)
    elif key == ord('1'):
        color_ranges['red']['enabled'] = not color_ranges['red']['enabled'] # Toggle red
        print(f"Red detection: {'ON' if color_ranges['red']['enabled'] else 'OFF'}")
    elif key == ord('2'):
        color_ranges['blue']['enabled'] = not color_ranges['blue']['enabled']
        print(f"Blue detection: {'ON' if color_ranges['blue']['enabled'] else 'OFF'}")
    elif key == ord('3'):
        color_ranges['green']['enabled'] = not color_ranges['green']['enabled']
        print(f"Green detection: {'ON' if color_ranges['green']['enabled'] else 'OFF'}")
    elif key == ord('4'):
        color_ranges['yellow']['enabled'] = not color_ranges['yellow']['enabled']
        print(f"Yellow detection: {'ON' if color_ranges['yellow']['enabled'] else 'OFF'}")
    elif key == ord('5'):
        color_ranges['orange']['enabled'] = not color_ranges['orange']['enabled']
        print(f"Orange detection: {'ON' if color_ranges['orange']['enabled'] else 'OFF'}")
    elif key == ord('6'):
        color_ranges['purple']['enabled'] = not color_ranges['purple']['enabled']
        print(f"Purple detection: {'ON' if color_ranges['purple']['enabled'] else 'OFF'}")
    
    # Select color for HSV tuning (letter keys)
    elif key == ord('r'):
        current_tuning_color = 'red'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: RED")
    elif key == ord('b'):
        current_tuning_color = 'blue'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: BLUE")
    elif key == ord('g'):
        current_tuning_color = 'green'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: GREEN")
    elif key == ord('y'):
        current_tuning_color = 'yellow'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: YELLOW")
    elif key == ord('o'):
        current_tuning_color = 'orange'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: ORANGE")
    elif key == ord('p'):
        current_tuning_color = 'purple'
        load_color_to_trackbars(current_tuning_color)
        print(f"Now tuning: PURPLE")
    
    # Get tuning status and trackbar values
    tuning_enabled = True
    
    if tuning_enabled:
        # Get current trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuning Controls')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuning Controls')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuning Controls')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuning Controls')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuning Controls')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuning Controls')
        
        # Update the currently selected color's HSV range
        update_color_range(current_tuning_color, h_min, s_min, v_min, h_max, s_max, v_max)
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Initialize frames for display
    contour_frame = frame.copy()
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    all_detected_objects = []
    
    # Process each enabled color
    for color_name, color_info in color_ranges.items():
        if color_info['enabled']:
            detected_objects, color_mask = detect_objects_by_color(hsv, frame, color_name, color_info)
            all_detected_objects.extend(detected_objects)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
    
    # Draw all detected objects
    for obj in all_detected_objects:
        x, y, w, h = obj['bbox']
        cx, cy = obj['center']
        draw_color = obj['draw_color']
        
        # Draw contour outline
        cv2.drawContours(contour_frame, [obj['contour']], -1, (0, 255, 0), 1)
        
        # Draw approximated polygon with color-specific color
        cv2.drawContours(contour_frame, [obj['approx']], -1, draw_color, 3)
        
        # Draw bounding rectangle with color-specific color
        cv2.rectangle(contour_frame, (x, y), (x + w, y + h), draw_color, 2)
        
        # Draw center point
        cv2.circle(contour_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Add intelligent label with background
        label = obj['label']
        label_size = cv2.getTextSize(label, DEFAULT_FONT, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(contour_frame, (x, y-40), (x + label_size[0] + 10, y-10), draw_color, -1)
        
        # Draw label text
        cv2.putText(contour_frame, label, (x+5, y-20), 
                   DEFAULT_FONT, 0.5, (255, 255, 255), 1)
        
        # Add detailed info
        cv2.putText(contour_frame, f"Area: {int(obj['area'])}", (x, y+h+15), 
                   DEFAULT_FONT, 0.35, draw_color, 1)
    
    # Create result frame with only detected colors
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # Add status information to the original frame
    frame_with_info = frame.copy()
    y_offset = 30
    
    # Show tuning status
    tuning_status = "ON" if tuning_enabled else "OFF"
    tuning_color = color_ranges[current_tuning_color]['bgr_color'] if tuning_enabled else (128, 128, 128)
    cv2.putText(frame_with_info, f"HSV Tuning: {tuning_status} ({current_tuning_color.title()})", (10, y_offset), 
               DEFAULT_FONT, 0.6, tuning_color, 2)
    y_offset += 30
    
    # Show current HSV values for tuning color if enabled
    if tuning_enabled:
        current_color = color_ranges[current_tuning_color]
        lower = current_color['lower1']
        upper = current_color['upper1']
        cv2.putText(frame_with_info, f"HSV: [{lower[0]},{lower[1]},{lower[2]}] to [{upper[0]},{upper[1]},{upper[2]}]", 
                   (10, y_offset), DEFAULT_FONT, 0.45, tuning_color, 1)
        y_offset += 25
    
    y_offset += 10  # Add some spacing
    
    # Show enabled colors and detection counts
    for color_name, color_info in color_ranges.items():
        status = "ON" if color_info['enabled'] else "OFF"
        color_count = len([obj for obj in all_detected_objects if obj['color'] == color_name])
        
        # Highlight the currently tuning color
        if color_name == current_tuning_color and tuning_enabled:
            text = f"{color_name.title()}: {status} ({color_count}) [TUNING]"
            thickness = 3
        else:
            text = f"{color_name.title()}: {status} ({color_count})"
            thickness = 2
            
        color = color_info['bgr_color'] if color_info['enabled'] else (128, 128, 128)
        
        cv2.putText(frame_with_info, text, (10, y_offset), 
                   DEFAULT_FONT, 0.5, color, thickness)
        y_offset += 25
    
    # Add total detection count
    total_objects = len(all_detected_objects)
    cv2.putText(frame_with_info, f"Total Objects: {total_objects}", (10, y_offset + 10), 
               DEFAULT_FONT, 0.6, (255, 255, 255), 2)
    
    # Add control instructions
    cv2.putText(frame_with_info, "1-6: Toggle colors | R,B,G,Y,O,P: Select tuning color | Q: Quit", 
               (10, frame.shape[0] - 40), DEFAULT_FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(frame_with_info, "Use HSV sliders to tune the selected color in real-time", 
               (10, frame.shape[0] - 20), DEFAULT_FONT, 0.45, (255, 255, 255), 1)
    
    # Create tuning visualization if tuning is enabled
    if tuning_enabled:
        # Create a mask showing only the currently tuning color for easier adjustment
        tuning_objects, tuning_mask = detect_objects_by_color(hsv, frame, current_tuning_color, color_ranges[current_tuning_color])
        tuning_result = cv2.bitwise_and(frame, frame, mask=tuning_mask)
        cv2.imshow(f"Tuning View - {current_tuning_color.title()}", tuning_result)
        cv2.imshow(f"Tuning Mask - {current_tuning_color.title()}", tuning_mask)
    else:
        # Close tuning windows when tuning is disabled
        cv2.destroyWindow(f"Tuning View - {current_tuning_color.title()}")
        cv2.destroyWindow(f"Tuning Mask - {current_tuning_color.title()}")
    
    # Display the main images
    cv2.imshow('HSV Tuning Controls', quit_box)
    cv2.imshow("Multi-Color Detection", frame_with_info)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Filtered Result", result)
    cv2.imshow("Labeled Objects", contour_frame)
    
    # Print detection summary and tuning info
    if tuning_enabled and total_objects > 0:
        tuning_count = len([obj for obj in all_detected_objects if obj['color'] == current_tuning_color])
        detected_labels = [obj['label'] for obj in all_detected_objects]
        print(f"Tuning {current_tuning_color.title()}: {tuning_count} detected | Total: {', '.join(detected_labels)}")
    elif total_objects > 0:
        detected_labels = [obj['label'] for obj in all_detected_objects]
        print(f"Detected: {', '.join(detected_labels)}")

cap.release()
cv2.destroyAllWindows()
