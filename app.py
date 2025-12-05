from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
from scipy import signal
from scipy.fft import fft
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import json

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        Image as RLImage,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = Flask(_name_)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Session storage (in production, use Redis or database)
sessions = {}


class VitalMonitorSession:
    def _init_(self, session_id):
        self.session_id = session_id
        self.ppg_signal = deque(maxlen=900)
        self.timestamps = deque(maxlen=900)
        self.hr_values = deque(maxlen=300)
        self.br_values = deque(maxlen=300)
        self.hrv_values = deque(maxlen=300)
        self.stress_values = deque(maxlen=300)
        self.para_values = deque(maxlen=300)
        self.wellness_values = deque(maxlen=300)
        self.bp_sys_values = deque(maxlen=300)
        self.bp_dia_values = deque(maxlen=300)

        self.results = {
            "heart_rate": 0,
            "breathing_rate": 0,
            "blood_pressure_sys": 0,
            "blood_pressure_dia": 0,
            "hrv": 0,
            "stress_index": 0,
            "parasympathetic": 0,
            "wellness_score": 0,
        }

        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "measurements": [],
        }

        self.frame_count = 0
        self.last_calculation_frame = 0
        # self.calculation_count = 0


def extract_ppg_signal(frame, landmarks):
    try:
        h, w = frame.shape[:2]

        forehead_indices = [10, 151, 9, 10, 151, 9, 10, 151]
        left_cheek_indices = [116, 117, 118, 119, 120, 121]
        right_cheek_indices = [345, 346, 347, 348, 349, 350]

        roi_values = []

        for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
            region_points = []
            for idx in indices:
                if idx < len(landmarks):
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    region_points.append([x, y])

            if len(region_points) > 2:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(region_points)], 255)
                green_channel = frame[:, :, 1]
                roi_mean = cv2.mean(green_channel, mask)[0]
                roi_values.append(roi_mean)

        if roi_values:
            return np.mean(roi_values)
        return 0
    except:
        return 0


def calculate_heart_rate(signal_data, fps=30):
    if len(signal_data) < fps * 8:
        return 0

    try:
        detrended = signal.detrend(signal_data)
        nyquist = fps / 2
        low = 0.8 / nyquist
        high = 4.0 / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, detrended)

        fft_data = fft(filtered)
        freqs = np.fft.fftfreq(len(filtered), 1 / fps)

        valid_indices = (freqs >= 0.8) & (freqs <= 4.0)
        valid_fft = np.abs(fft_data[valid_indices])
        valid_freqs = freqs[valid_indices]

        if len(valid_fft) > 0:
            peak_idx = np.argmax(valid_fft)
            heart_rate_hz = valid_freqs[peak_idx]
            heart_rate_bpm = heart_rate_hz * 60
            return max(50, min(200, heart_rate_bpm))
    except:
        pass
    return 0


def calculate_breathing_rate(signal_data, fps=30):
    if len(signal_data) < fps * 12:
        return 0

    try:
        nyquist = fps / 2
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = signal.butter(2, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, signal_data)

        peaks, _ = signal.find_peaks(filtered, distance=fps * 2)
        breathing_rate = len(peaks) * (60 / (len(signal_data) / fps))
        return max(8, min(35, breathing_rate))
    except:
        return 0


def calculate_hrv(signal_data, fps=30):
    if len(signal_data) < fps * 15:
        return 0

    try:
        filtered = signal.medfilt(signal_data, 5)
        peaks, _ = signal.find_peaks(filtered, distance=fps // 3)

        if len(peaks) < 5:
            return 0

        intervals = np.diff(peaks) / fps * 1000
        successive_diffs = np.diff(intervals)
        rmssd = np.sqrt(np.mean(successive_diffs**2))
        return min(100, max(10, rmssd))
    except:
        return 0


def calculate_stress_index(heart_rate, hrv, breathing_rate):
    try:
        hr_stress = max(0, (heart_rate - 70) / 50)
        hrv_stress = max(0, (50 - hrv) / 50)
        br_stress = max(0, (breathing_rate - 15) / 15)
        stress_index = (hr_stress + hrv_stress + br_stress) / 3
        return min(1.0, max(0.0, stress_index))
    except:
        return 0


def calculate_parasympathetic_activity(hrv, breathing_rate):
    try:
        hrv_factor = min(1.0, hrv / 50)
        breathing_factor = max(0, (20 - breathing_rate) / 10)
        parasympathetic = (hrv_factor + breathing_factor) / 2 * 100
        return min(100, max(0, parasympathetic))
    except:
        return 50


def estimate_blood_pressure(heart_rate, hrv, stress_index):
    try:
        base_sys = 120
        base_dia = 80

        hr_factor = (heart_rate - 70) * 0.5
        stress_factor = stress_index * 10
        hrv_factor = (50 - hrv) * 0.2

        sys_bp = base_sys + hr_factor + stress_factor + hrv_factor
        dia_bp = base_dia + hr_factor * 0.6 + stress_factor * 0.6 + hrv_factor * 0.6

        sys_bp = max(90, min(180, sys_bp))
        dia_bp = max(60, min(120, dia_bp))

        return int(sys_bp), int(dia_bp)
    except:
        return 120, 80


def calculate_wellness_score(results):
    try:
        hr = results["heart_rate"]
        hrv = results["hrv"]
        stress = results["stress_index"]
        para = results["parasympathetic"]

        hr_score = 1 - abs(hr - 70) / 50 if hr > 0 else 0.5
        hrv_score = min(1, hrv / 50)
        stress_score = 1 - stress
        para_score = para / 100

        wellness = (hr_score + hrv_score + stress_score + para_score) / 4 * 100
        return max(0, min(100, wellness))
    except:
        return 50


# def get_status_text(metric, results):
#     if metric == "heart_rate":
#         hr = results["heart_rate"]
#         if 60 <= hr <= 100:
#             return "Normal"
#         elif 50 <= hr <= 120:
#             return "Acceptable"
#         else:
#             return "Abnormal"
#     elif metric == "breathing_rate":
#         br = results["breathing_rate"]
#         if 12 <= br <= 20:
#             return "Normal"
#         elif 8 <= br <= 25:
#             return "Acceptable"
#         else:
#             return "Abnormal"
#     elif metric == "blood_pressure":
#         sys_bp = results["blood_pressure_sys"]
#         dia_bp = results["blood_pressure_dia"]
#         if sys_bp < 130 and dia_bp < 85:
#             return "Normal"
#         elif sys_bp < 140 and dia_bp < 90:
#             return "Elevated"
#         else:
#             return "High"
#     elif metric == "hrv":
#         hrv = results["hrv"]
#         if hrv > 40:
#             return "Good"
#         elif hrv > 20:
#             return "Fair"
#         else:
#             return "Poor"
#     elif metric == "stress_index":
#         stress = results["stress_index"]
#         if stress < 0.3:
#             return "Low"
#         elif stress < 0.7:
#             return "Moderate"
#         else:
#             return "High"
#     elif metric == "parasympathetic":
#         para = results["parasympathetic"]
#         if para > 60:
#             return "Good"
#         elif para > 30:
#             return "Fair"
#         else:
#             return "Poor"
#     elif metric == "wellness_score":
#         wellness = results["wellness_score"]
#         if wellness > 70:
#             return "Excellent"
#         elif wellness > 40:
#             return "Good"
#         else:
#             return "Needs Improvement"
#     return "Unknown"


# def get_health_interpretation(results):
#     interpretations = []

#     hr = results["heart_rate"]
#     if hr < 60:
#         interpretations.append(
#             "• Heart Rate: Below normal (Bradycardia) - Consider consulting a healthcare provider."
#         )
#     elif hr > 100:
#         interpretations.append(
#             "• Heart Rate: Above normal (Tachycardia) - May indicate stress, exercise, or other factors."
#         )
#     else:
#         interpretations.append("• Heart Rate: Normal range (60-100 bpm).")

#     sys_bp = results["blood_pressure_sys"]
#     dia_bp = results["blood_pressure_dia"]
#     if sys_bp < 120 and dia_bp < 80:
#         interpretations.append("• Blood Pressure: Normal (<120/80 mmHg).")
#     elif sys_bp < 130 and dia_bp < 85:
#         interpretations.append("• Blood Pressure: Normal to slightly elevated.")
#     elif sys_bp < 140 and dia_bp < 90:
#         interpretations.append(
#             "• Blood Pressure: Stage 1 hypertension - monitor regularly."
#         )
#     else:
#         interpretations.append(
#             "• Blood Pressure: High (≥140/90 mmHg) - Recommend medical evaluation."
#         )

#     hrv = results["hrv"]
#     if hrv > 40:
#         interpretations.append(
#             "• Heart Rate Variability: Good - indicates healthy autonomic nervous system."
#         )
#     elif hrv > 20:
#         interpretations.append(
#             "• Heart Rate Variability: Fair - room for improvement through stress management."
#         )
#     else:
#         interpretations.append(
#             "• Heart Rate Variability: Low - may indicate stress or autonomic dysfunction."
#         )

#     stress = results["stress_index"]
#     if stress < 0.3:
#         interpretations.append(
#             "• Stress Level: Low - maintaining good stress management."
#         )
#     elif stress < 0.7:
#         interpretations.append(
#             "• Stress Level: Moderate - consider stress reduction techniques."
#         )
#     else:
#         interpretations.append(
#             "• Stress Level: High - recommend stress management and relaxation practices."
#         )

#     wellness = results["wellness_score"]
#     if wellness > 70:
#         interpretations.append(
#             "• Overall Wellness: Excellent - maintaining good health habits."
#         )
#     elif wellness > 40:
#         interpretations.append(
#             "• Overall Wellness: Good - some areas for improvement identified."
#         )
#     else:
#         interpretations.append(
#             "• Overall Wellness: Needs attention - consider comprehensive health evaluation."
#         )

#     return "\n".join(interpretations)


# def get_recommendations(results):
#     recommendations = []

#     hr = results["heart_rate"]
#     stress = results["stress_index"]
#     hrv = results["hrv"]
#     wellness = results["wellness_score"]

#     recommendations.append("GENERAL RECOMMENDATIONS:")
#     recommendations.append(
#         "• Maintain regular exercise routine (150 minutes moderate activity per week)"
#     )
#     recommendations.append(
#         "• Practice stress management techniques (meditation, deep breathing)"
#     )
#     recommendations.append("• Ensure adequate sleep (7-9 hours per night)")
#     recommendations.append("• Stay hydrated and maintain balanced nutrition")

#     if stress > 0.5:
#         recommendations.append("\nSTRESS MANAGEMENT:")
#         recommendations.append("• Consider mindfulness meditation or yoga")
#         recommendations.append("• Practice progressive muscle relaxation")
#         recommendations.append("• Limit caffeine intake")
#         recommendations.append("• Ensure work-life balance")

#     if hrv < 30:
#         recommendations.append("\nHEART RATE VARIABILITY IMPROVEMENT:")
#         recommendations.append("• Regular cardiovascular exercise")
#         recommendations.append("• Breathing exercises (4-7-8 technique)")
#         recommendations.append("• Reduce alcohol consumption")
#         recommendations.append("• Consider heart rate variability training")

#     if hr > 100 or hr < 60:
#         recommendations.append("\nHEART RATE CONCERNS:")
#         recommendations.append("• Monitor heart rate regularly")
#         recommendations.append("• Consult healthcare provider if persistent")
#         recommendations.append("• Avoid excessive caffeine")
#         recommendations.append("• Maintain regular sleep schedule")

#     if wellness < 50:
#         recommendations.append("\nWELLNESS IMPROVEMENT:")
#         recommendations.append("• Comprehensive health assessment recommended")
#         recommendations.append("• Consider lifestyle modifications")
#         recommendations.append("• Regular monitoring of vital signs")
#         recommendations.append("• Professional health consultation advised")

#     recommendations.append("\nDISCLAIMER:")
#     recommendations.append("This is a demonstration tool for educational purposes.")
#     recommendations.append(
#         "Measurements are estimates and should not replace professional medical advice."
#     )
#     recommendations.append("Consult healthcare professionals for medical concerns.")

#     return "\n".join(recommendations)


def calculate_all_metrics(session):
    if len(session.ppg_signal) < 300:
        return

    signal_array = np.array(list(session.ppg_signal))

    hr = calculate_heart_rate(signal_array)
    session.results["heart_rate"] = int(hr) if hr > 0 else 0
    session.hr_values.append(session.results["heart_rate"])

    br = calculate_breathing_rate(signal_array)
    session.results["breathing_rate"] = int(br) if br > 0 else 0
    session.br_values.append(session.results["breathing_rate"])

    hrv = calculate_hrv(signal_array)
    session.results["hrv"] = int(hrv)
    session.hrv_values.append(session.results["hrv"])

    stress = calculate_stress_index(hr, hrv, br)
    session.results["stress_index"] = round(stress, 2)
    session.stress_values.append(session.results["stress_index"])

    para = calculate_parasympathetic_activity(hrv, br)
    session.results["parasympathetic"] = int(para)
    session.para_values.append(session.results["parasympathetic"])

    sys_bp, dia_bp = estimate_blood_pressure(hr, hrv, stress)
    session.results["blood_pressure_sys"] = sys_bp
    session.results["blood_pressure_dia"] = dia_bp
    session.bp_sys_values.append(sys_bp)
    session.bp_dia_values.append(dia_bp)

    wellness = calculate_wellness_score(session.results)
    session.results["wellness_score"] = int(wellness)
    session.wellness_values.append(session.results["wellness_score"])

    measurement = {"timestamp": datetime.now().isoformat(), **session.results}
    session.session_data["measurements"].append(measurement)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.json
        session_id = data.get("session_id", "default")
        image_data = data["image"].split(",")[1]
        monitoring = data.get("monitoring", False)

        # Get or create session
        if session_id not in sessions:
            sessions[session_id] = VitalMonitorSession(session_id)

        session = sessions[session_id]

        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]

            # Draw styled mesh (matching Streamlit version)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=1
                ),
            )
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=1
                ),
            )

            # Extract PPG signal if monitoring
            if monitoring:
                ppg_value = extract_ppg_signal(frame, face_landmarks.landmark)
                current_time = datetime.now().timestamp()

                session.ppg_signal.append(ppg_value)
                session.timestamps.append(current_time)
                session.frame_count += 1

                # Calculate metrics every 30 frames (once per second at 30fps)
                if (
                    session.frame_count >= 150
                    and (session.frame_count - session.last_calculation_frame) >= 30
                ):
                    calculate_all_metrics(session)
                    session.last_calculation_frame = session.frame_count

        # Convert frame back to base64
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG", quality=85)
        img_str = base64.b64encode(buff.getvalue()).decode()

        return jsonify(
            {
                "success": True,
                "face_detected": face_detected,
                "processed_image": f"data:image/jpeg;base64,{img_str}",
                "results": session.results,
                "signal_length": len(session.ppg_signal),
                "frame_count": session.frame_count,
                "monitoring_complete": len(session.ppg_signal) >= 900,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_trends/<session_id>")
def get_trends(session_id):
    try:
        if session_id not in sessions:
            return jsonify({"success": False, "error": "Session not found"})

        session = sessions[session_id]

        return jsonify(
            {
                "success": True,
                "hr_values": list(session.hr_values),
                "br_values": list(session.br_values),
                "hrv_values": list(session.hrv_values),
                "stress_values": list(session.stress_values),
                "para_values": list(session.para_values),
                "wellness_values": list(session.wellness_values),
                "bp_sys_values": list(session.bp_sys_values),
                "bp_dia_values": list(session.bp_dia_values),
                "ppg_signal": list(session.ppg_signal),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/reset_session/<session_id>", methods=["POST"])
def reset_session(session_id):
    try:
        if session_id in sessions:
            del sessions[session_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/generate_report/<session_id>")
def generate_report(session_id):
    try:
        if session_id not in sessions:
            return jsonify({"success": False, "error": "Session not found"})

        if not PDF_AVAILABLE:
            return jsonify({"success": False, "error": "PDF generation not available"})

        session = sessions[session_id]

        if not session.session_data["measurements"]:
            return jsonify({"success": False, "error": "No measurements available"})

        # Generate PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=20,
            spaceAfter=30,
            alignment=1,
        )
        story.append(Paragraph("COMPREHENSIVE HEALTH MONITORING REPORT", title_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("FINAL HEALTH MEASUREMENTS", styles["Heading2"]))
        measurements_data = [
            ["Metric", "Value"],
            ["Heart Rate", f"{session.results['heart_rate']} bpm"],
            ["Breathing Rate", f"{session.results['breathing_rate']} rpm"],
            [
                "Blood Pressure",
                f"{session.results['blood_pressure_sys']}/{session.results['blood_pressure_dia']} mmHg",
            ],
            ["HRV", f"{session.results['hrv']} ms"],
            ["Stress Index", f"{session.results['stress_index']}"],
            ["Parasympathetic Activity", f"{session.results['parasympathetic']}%"],
            ["Wellness Score", f"{session.results['wellness_score']}/100"],
        ]
        measurements_table = Table(measurements_data, colWidths=[3 * inch, 3 * inch])
        measurements_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(measurements_table)
        doc.build(story)

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f'Health_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
