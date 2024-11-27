import cv2

# Load Haar Cascades untuk deteksi wajah dan mata
face_ref = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_ref = cv2.CascadeClassifier("haarcascade_eye.xml")

# Validasi apakah file XML tersedia
if face_ref.empty() or eye_ref.empty():
    print("Error: Haar Cascade XML tidak ditemukan atau tidak valid.")
    exit()

# Inisialisasi kamera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 840)  # Atur resolusi kamera
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

def detect_features(frame):
    """
    Fungsi untuk mendeteksi wajah dan mata pada frame.
    """
    # Konversi frame ke grayscale untuk deteksi
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_ref.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Wajah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ROI (Region of Interest) untuk mata
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Deteksi mata di dalam wajah
        eyes = eye_ref.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Mata", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def close_window():
    """
    Fungsi untuk menutup kamera dan jendela OpenCV.
    """
    camera.release()
    cv2.destroyAllWindows()

def main():
    """
    Fungsi utama untuk menjalankan program deteksi wajah dan mata.
    """
    while True:
        # Baca frame dari kamera
        ret, frame = camera.read()
        if not ret:
            print("Tidak dapat membaca frame dari kamera.")
            break

        # Deteksi wajah dan mata
        detect_features(frame)

        # Tampilkan frame dengan deteksi fitur
        cv2.imshow("Face and Eye Detection", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
            break

if __name__ == "__main__":
    main()
