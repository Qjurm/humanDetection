from http import cookies

import cv2
import requests
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# --- Instellingen ---
CSV_FILENAME = "demo.csv"
data = 0

# --- Globale variabelen ---
address = "https://192.168.12.1:5173/api/runs/-1/submissions/-1"
drawing = False
ix, iy = -1, -1
zones = []  # Lijst: [(x1, y1, x2, y2), ...]

# Tracking data
visited_ids = {}  # {zone_index: {id1, id2, ...}}
current_counts = {}  # {zone_index: int}


# --- Muis Functie ---
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, zones, visited_ids, current_counts

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)

        # Filter te kleine klikjes eruit
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            zones.append((x1, y1, x2, y2))
            idx = len(zones) - 1
            visited_ids[idx] = set()
            current_counts[idx] = 0
            print(f"Zone {idx + 1} toegevoegd.")


# --- CSV Functie (Direct Opslaan) ---
def save_event_to_csv(zone_index):
    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Header schrijven als bestand nog niet bestaat
        if not file_exists:
            writer.writerow(["Tijdstip", "Zone_ID", "Aantal_Mensen"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # We schrijven 1 regel per detectie
        # "1" in de laatste kolom betekent: 1 persoon gedetecteerd op dit moment
        writer.writerow([timestamp, f"Zone {zone_index + 1}", 1])

    print(f"-> Opgeslagen: Zone {zone_index + 1} om {timestamp}")


# --- Setup ---
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)  # Let op: check of dit camera index 2, 0 of 1 moet zijn

window_name = "YOLO People Counter (Live Save)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_rectangle)

print(f"Systeem gestart. Detectie wordt DIRECT opgeslagen in CSV.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking aanzetten
    results = model.track(frame, persist=True, verbose=False, classes=[0])

    # Resultaten verwerken
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # --- AANPASSING: BEREKEN ONDERKANT ---
            center_x = int((x1 + x2) / 2)
            center_y = y2  # We gebruiken de voeten/onderkant
            # --------------------------------------

            # Teken box en ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Teken de 'trigger' punt
            cv2.circle(frame, (center_x, center_y), 6, (255, 0, 255), -1)

            # Check of MIDDELPUNT in een zone valt
            for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
                if i not in visited_ids: visited_ids[i] = set()
                if i not in current_counts: current_counts[i] = 0

                # Checken of punt in de zone is
                if zx1 < center_x < zx2 and zy1 < center_y < zy2:
                    # Check unieke ID (om dubbele tellingen te voorkomen)
                    if track_id not in visited_ids[i]:
                        visited_ids[i].add(track_id)
                        current_counts[i] += 1

                        # --- HIER WORDT DIRECT OPGESLAGEN ---
                        save_event_to_csv(i)
                        # ------------------------------------
                        #send to DB
                        doorNMR = 0
                        if i == 0:
                            doorNMR = -1
                        else:
                            doorNMR = 1

                        json = {"doorStatus": doorNMR, "automated": True}

                        print(doorNMR)

                        # requests.patch(url=address, json=json, cookies={"observerpwrd":"Kien"}, verify=False)

                        print(f"ID {track_id} geteld in Zone {i + 1}")

    # --- Tekenen van UI ---
    # Zones tekenen
    for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
        count = current_counts.get(i, 0)
        # Blauwe rand, tekst erin
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 100, 0), 2)
        cv2.putText(frame, f"Z{i + 1}: {count}", (zx1, zy1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Live Logging Active", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Reset de huidige zichtbare tellers en ID geheugen
        # Let op: dit wist NIET de CSV, alleen het geheugen van het script
        zones = []
        visited_ids = {}
        current_counts = {}
        print("Zones en lokaal geheugen gewist.")

cap.release()
cv2.destroyAllWindows()