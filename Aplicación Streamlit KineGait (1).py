import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="KineGait - Análisis de Marcha Biomecánico",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 12px;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 20px;
        border: 1px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# INICIALIZACIÓN DE MEDIAPIPE (COMPATIBILIDAD ROBUSTA)
# ==============================================================================
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# ==============================================================================
# FUNCIONES AUXILIARES DE BIOMECÁNICA
# ==============================================================================
def calculate_angle(a, b, c):
    """
    Calcula el ángulo en grados entre tres puntos (a: origen/proximal, b: articulación, c: distal).
    Utiliza arctan2 para estabilidad numérica en 2D.
    """
    a = np.array(a) # Ejemplo: Cadera [x, y]
    b = np.array(b) # Ejemplo: Rodilla [x, y]
    c = np.array(c) # Ejemplo: Tobillo [x, y]
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def calculate_tilt_angle(p1, p2):
    """
    Calcula el ángulo de inclinación de un segmento (p1-p2) respecto a la horizontal.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = np.abs(np.arctan2(delta_y, delta_x) * 180.0 / np.pi)
    return angle

def calculate_symmetry_index(val_affected, val_sound):
    """
    Calcula el Índice de Simetría (%):
    IS = |Afectado - Sano| / (0.5 * (Afectado + Sano)) * 100
    """
    if val_affected is None or val_sound is None or (val_affected + val_sound) == 0:
        return 0.0
    return (abs(val_affected - val_sound) / (0.5 * (val_affected + val_sound))) * 100.0

# ==============================================================================
# INTERFAZ Y BARRA LATERAL
# ==============================================================================
st.markdown('<div class="main-header">KineGait 🏃‍♂️️ Core Biomechanic Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Plataforma clínica para cuantificación de marcha y simetría en rehabilitación</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <b>Aviso de uso clínico:</b> Herramienta educativa de apoyo y tamizaje biomecánico en tiempo real; 
    no reemplaza una evaluación clínica formal ni un sistema optoelectrónico de captura de movimiento tridimensional (ej. Vicon). 
    <i>Todos los datos se procesan localmente sin guardar copias en servidores externos.</i>
</div>
""", unsafe_allow_html=True)

# Configuración en la barra lateral
st.sidebar.header("⚙️ Configuración del Análisis")

plane = st.sidebar.selectbox(
    "Plano de Observación",
    ["Sagital (Perfil)", "Frontal (Anterior/Posterior)"],
    help="Sagital: flexión de rodilla, inclinación de tronco y longitud de paso. Frontal: valgo/varo y báscula pélvica."
)

affected_side = st.sidebar.radio(
    "Lado Afectado / En Rehabilitación",
    ["Izquierdo", "Derecho"],
    index=1
)

user_height_cm = st.sidebar.number_input(
    "Estatura del Paciente (cm)",
    min_value=100.0,
    max_value=230.0,
    value=170.0,
    step=1.0,
    help="Utilizado para la calibración antropométrica de longitud de paso (px a metros)."
)

source_option = st.sidebar.radio(
    "Fuente de Entrada de Video",
    ["Cámara en Vivo", "Subir Archivo de Video"]
)

# ==============================================================================
# PROCESAMIENTO PRINCIPAL DE VIDEO Y CAPTURA DE MARCHA
# ==============================================================================
col_video, col_metrics = st.columns([2, 1])

with col_metrics:
    st.subheader("📊 Muestreo en Vivo")
    metric_knee_left = st.empty()
    metric_knee_right = st.empty()
    metric_trunk = st.empty()
    metric_fps = st.empty()

# Contenedor para el stream de video
video_placeholder = col_video.empty()

# Listas de registro temporal
time_stamps = []
knee_angles_left = []
knee_angles_right = []
trunk_angles = []
heel_distances = []

# Variables biomecánicas acumulativas
max_flex_left = 0.0
max_flex_right = 0.0
min_flex_left = 180.0
min_flex_right = 180.0
num_steps = 0

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

video_capture = None
is_running = False

if source_option == "Cámara en Vivo":
    st.sidebar.info("Asegúrate de que la cámara esté fija a la altura de la cadera y perpendicular al recorrido del paciente.")
    start_cam = st.sidebar.checkbox("Activar Cámara en Vivo")
    if start_cam:
        video_capture = cv2.VideoCapture(0)
        is_running = True
else:
    uploaded_file = st.sidebar.file_uploader("Selecciona un video de marcha (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_capture = cv2.VideoCapture(tfile.name)
        is_running = True

# Bucle de procesamiento frame a frame
if is_running and video_capture is not None and video_capture.isOpened():
    start_time = time.time()
    frame_count = 0
    scale_factor = None # px a metros

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        current_timestamp = time.time() - start_time
        
        # Conversión de color para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Puntos clave biomecánicos
            lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
            la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
            lhe = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * height]

            rh = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
            rk = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
            ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]
            rhe = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * height]

            ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]

            # Calibración inicial aproximada (Estatura en px)
            if scale_factor is None:
                head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * height
                feet_y = max(la[1], ra[1])
                body_px_height = abs(feet_y - head_y)
                if body_px_height > 100:
                    scale_factor = (user_height_cm / 100.0) / body_px_height # Metros por píxel

            # Cálculos en Plano Sagital
            if "Sagital" in plane:
                angle_k_left = calculate_angle(lh, lk, la)
                angle_k_right = calculate_angle(rh, rk, ra)
                
                # Inclinación de tronco respecto a la vertical
                mid_hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
                mid_shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
                trunk_angle = calculate_angle([mid_hip[0], mid_hip[1] - 100], mid_hip, mid_shoulder)

                # Guardar valores
                knee_angles_left.append(angle_k_left)
                knee_angles_right.append(angle_k_right)
                trunk_angles.append(trunk_angle)
                time_stamps.append(current_timestamp)

                max_flex_left = max(max_flex_left, angle_k_left)
                max_flex_right = max(max_flex_right, angle_k_right)
                min_flex_left = min(min_flex_left, angle_k_left)
                min_flex_right = min(min_flex_right, angle_k_right)

                # Detección de contacto inicial por separación horizontal de talones
                heel_dist = abs(lhe[0] - rhe[0])
                heel_distances.append(heel_dist)

                # Renderizado sobre el video
                left_color = (0, 0, 255) if affected_side == "Izquierdo" else (0, 255, 0)
                right_color = (0, 0, 255) if affected_side == "Derecho" else (0, 255, 0)

                cv2.line(frame, (int(lh[0]), int(lh[1])), (int(lk[0]), int(lk[1])), left_color, 4)
                cv2.line(frame, (int(lk[0]), int(lk[1])), (int(la[0]), int(la[1])), left_color, 4)
                cv2.line(frame, (int(rh[0]), int(rh[1])), (int(rk[0]), int(rk[1])), right_color, 4)
                cv2.line(frame, (int(rk[0]), int(rk[1])), (int(ra[0]), int(ra[1])), right_color, 4)

                metric_knee_left.metric("Ángulo Rodilla Izq.", f"{angle_k_left:.1f}°")
                metric_knee_right.metric("Ángulo Rodilla Der.", f"{angle_k_right:.1f}°")
                metric_trunk.metric("Inclinación Tronco", f"{trunk_angle:.1f}°")

            else:
                # Plano Frontal (Alineación / Báscula)
                pelvis_tilt = calculate_tilt_angle(lh, rh)
                shoulder_tilt = calculate_tilt_angle(ls, rs)
                knee_valgus_left = calculate_angle(lh, lk, la)
                knee_valgus_right = calculate_angle(rh, rk, ra)

                metric_knee_left.metric("Alineación Rodilla Izq.", f"{knee_valgus_left:.1f}°")
                metric_knee_right.metric("Alineación Rodilla Der.", f"{knee_valgus_right:.1f}°")
                metric_trunk.metric("Inclinación Pelvis", f"{pelvis_tilt:.1f}°")

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Cálculo de FPS
            fps = frame_count / (time.time() - start_time)
            metric_fps.metric("Velocidad Procesamiento", f"{fps:.1f} FPS")

        # Convertir a RGB para mostrar en Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    video_capture.release()

# ==============================================================================
# INFORME CLÍNICO DE RESULTADOS E ÍNDICES DE SIMETRÍA
# ==============================================================================
if len(time_stamps) > 10:
    st.markdown("---")
    st.header("📋 Informe Biomecánico y Resumen de Sesión")

    # Cálculos temporales e Índices de Simetría
    val_aff_flex = max_flex_left if affected_side == "Izquierdo" else max_flex_right
    val_sound_flex = max_flex_right if affected_side == "Izquierdo" else max_flex_left
    si_flexion = calculate_symmetry_index(val_aff_flex, val_sound_flex)

    # Estimación de cadencia (pasos por minuto) mediante picos de separación de talones
    if len(heel_distances) > 10:
        smoothed_dist = np.convolve(heel_distances, np.ones(5)/5, mode='same')
        peaks = [i for i in range(1, len(smoothed_dist)-1) if smoothed_dist[i-1] < smoothed_dist[i] > smoothed_dist[i+1] and smoothed_dist[i] > np.mean(smoothed_dist)]
        num_steps = len(peaks)
        total_duration_sec = time_stamps[-1] - time_stamps[0]
        cadence = (num_steps / total_duration_sec) * 60.0 if total_duration_sec > 0 else 0
    else:
        cadence = 0

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.subheader("⏱️ Parámetros Espaciotemporales")
        st.write(f"**Cadencia Estimada:** {cadence:.1f} pasos/min")
        st.write(f"**Duración Registrada:** {time_stamps[-1]:.2f} segundos")
        st.write(f"**Total de Contactos Detectados:** {num_steps} pasos")

    with col_res2:
        st.subheader("🦵 Rango de Flexión Máxima")
        st.write(f"**Lado Afectado ({affected_side}):** {val_aff_flex:.1f}°")
        st.write(f"**Lado Sano:** {val_sound_flex:.1f}°")
        st.write(f"**Flexión Mínima (Extensión):** Izq {min_flex_left:.1f}° | Der {min_flex_right:.1f}°")

    with col_res3:
        st.subheader("⚖️ Simetría Biomecánica")
        st.metric("Índice de Simetría (Flexión)", f"{si_flexion:.1f}%")
        
        if si_flexion < 10.0:
            st.success("🟢 Simetría Normal (Diferencia < 10%)")
        elif si_flexion <= 20.0:
            st.warning("🟡 Asimetría Leve-Moderada (10% - 20%)")
        else:
            st.error("🔴 Asimetría Severa (> 20%) - Sugiere compensación marcada")

    # Gráficos dinámicos con Plotly
    st.subheader("📈 Curva Cinemática de Flexo-Extensión de Rodilla")
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=time_stamps, y=knee_angles_left, mode='lines', name='Rodilla Izquierda', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time_stamps, y=knee_angles_right, mode='lines', name='Rodilla Derecha', line=dict(color='orange')))
    
    fig.update_layout(
        title="Ángulo de Flexión de Rodilla en el Tiempo (Grados vs Segundos)",
        xaxis_title="Tiempo (s)",
        yaxis_title="Ángulo de Rodilla (°)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Exportación de datos a CSV
    st.subheader("💾 Exportar Datos")
    df_export = pd.DataFrame({
        "Tiempo_s": time_stamps,
        "Angulo_Rodilla_Izq_deg": knee_angles_left,
        "Angulo_Rodilla_Der_deg": knee_angles_right,
        "Inclinacion_Tronco_deg": trunk_angles
    })
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar CSV con Registro Temporal del Análisis",
        data=csv_data,
        file_name="KineGait_analisis_marcha.csv",
        mime="text/csv"
    )
```
```text:Archivo de Dependencias:requirements.txt
streamlit>=1.30.0
mediapipe==0.10.14
protobuf>=3.20.3,<4.25.0
opencv-python-headless>=4.8.0.76
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
plotly>=5.18.0