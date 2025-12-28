from deepface import DeepFace
import os
import shutil

KNOWN_DIR = "known_faces"
ALL_DIR = "all_photos"
RESULT_DIR = "result"

# STRICT threshold (lower = more strict)
THRESHOLD = 0.35  

people = os.listdir(KNOWN_DIR)

print("Select a person:")
for i, p in enumerate(people):
    print(f"{i+1}. {p}")

choice = int(input("Enter number: ")) - 1
selected_person = people[choice]
ref_path = os.path.join(KNOWN_DIR, selected_person)

os.makedirs(RESULT_DIR, exist_ok=True)

for photo in os.listdir(ALL_DIR):
    photo_path = os.path.join(ALL_DIR, photo)

    try:
        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=photo_path,
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True
        )

        distance = result["distance"]

        if distance < THRESHOLD:
            shutil.copy(photo_path, os.path.join(RESULT_DIR, photo))
            print(f"✔ MATCH ({distance:.3f}): {photo}")
        else:
            print(f"✖ NO MATCH ({distance:.3f}): {photo}")

    except Exception as e:
        print("Skipped:", photo)

print("\nDone.")