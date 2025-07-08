    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    from datetime import datetime

    # Load remedies and product recommendations
    REMEDIES_PATH = r"C:\Users\shaik\Downloads\facialanalysisdataset.csv"
    # Load the dataset with the correct delimiter
    remedies_df = pd.read_csv(REMEDIES_PATH, sep=',', encoding='utf-8')  # Change sep to ';' if needed
    remedies_df.columns = remedies_df.columns.str.strip().str.lower().str.replace(" ", "_")
    print("Standardized column names:", remedies_df.columns.tolist())


    # Mediapipe for face detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def check_lighting(frame):
        """Check if the lighting is adequate based on brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return brightness > 100  # Threshold for adequate lighting

    def analyze_skin(frame, detections):
        """Analyze skin concerns from the detected face and return results in percentage."""
        conditions = {}
        for detection in detections:
            # Extract bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face = frame[y:y+h, x:x+w]
            hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)


            # Dark Circles (under eyes)
            under_eye_area = face[int(0.6 * h):h, :]
            under_eye_gray = cv2.cvtColor(under_eye_area, cv2.COLOR_BGR2GRAY)
            avg_darkness = 255 - np.mean(under_eye_gray)
            conditions["Dark Circles"] = f"{int((avg_darkness / 255) * 100)}%"

            # Acne (blemishes)
            acne_area = cv2.inRange(face, (0, 0, 0), (80, 80, 80))
            acne_count = cv2.countNonZero(acne_area)
            conditions["Acne"] = "Moderate" if acne_count > 100 else "Mild"

            # Redness (inflammation)
            redness_area = cv2.inRange(face, (0, 0, 100), (50, 50, 255))
            redness_count = cv2.countNonZero(redness_area)
            conditions["Redness"] = "High" if redness_count > 100 else "Low"

            # Dark Spots / Hyperpigmentation
            dark_spots_area = cv2.inRange(face, (0, 0, 50), (80, 80, 255))
            dark_spots_count = cv2.countNonZero(dark_spots_area)
            conditions["Dark Spots"] = f"{int((dark_spots_count / (w * h)) * 100)}%"

            # Melasma (usually on cheeks/forehead)
            melasma_area = cv2.inRange(face, (0, 50, 100), (80, 120, 200))
            melasma_count = cv2.countNonZero(melasma_area)
            conditions["Melasma"] = f"{int((melasma_count / (w * h)) * 100)}%"

            # Uneven Skin Tone
            skin_tone_variance = np.var(hsv_face[:, :, 2])  # Variance in brightness (value channel)
            conditions["Uneven Skin Tone"] = f"{int(min((skin_tone_variance / 255) * 100, 100))}%"

            # Large Pores (based on texture, size of visible pores)
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 100, 200)
            pore_area = cv2.countNonZero(edges)
            conditions["Large Pores"] = f"{int((pore_area / (w * h)) * 100)}%"

            # Rough Texture
            rough_texture_area = cv2.Laplacian(gray_face, cv2.CV_64F)
            rough_texture_area = np.uint8(np.absolute(rough_texture_area))  # Convert to uint8
            _, binary_roughness = cv2.threshold(rough_texture_area, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_roughness, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Bumpy Skin (using contour analysis)
            contours, _ = cv2.findContours(rough_texture_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bumpy_skin_count = len(contours)
            conditions["Bumpy Skin"] = f"{int((bumpy_skin_count / 50) * 100)}%"  # Example threshold

            # Dullness (based on brightness and color saturation)
            avg_brightness = np.mean(hsv_face[:, :, 2]) / 255  # Normalize to [0, 1] range
            avg_saturation = np.mean(hsv_face[:, :, 1]) / 255  # Normalize to [0, 1] range
            conditions["Dullness"] = f"{int(min(((avg_brightness + avg_saturation) / 2) * 100, 100))}%"

            # Tired-Looking Skin (based on brightness and skin tone)
            tired_skin_area = cv2.inRange(hsv_face, (0, 0, 50), (50, 50, 150))
            tired_skin_count = cv2.countNonZero(tired_skin_area)
            conditions["Tired-Looking Skin"] = f"{int((tired_skin_count / (w * h)) * 100)}%"

            # Eye Bags (below the eyes)
            under_eye_bags_area = face[int(0.7 * h):h, :]
            eye_bags_gray = cv2.cvtColor(under_eye_bags_area, cv2.COLOR_BGR2GRAY)
            avg_eye_bags = np.mean(eye_bags_gray)
            conditions["Eye Bags"] = f"{int((avg_eye_bags / 255) * 100)}%"

            # Fine Lines Around Eyes (using edge detection)
            eye_area = face[int(0.4 * h):int(0.6 * h), :]
            fine_lines_area = cv2.Laplacian(cv2.cvtColor(eye_area, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            fine_lines_count = cv2.countNonZero(fine_lines_area)
            conditions["Fine Lines Around Eyes"] = f"{int((fine_lines_count / (w * h)) * 100)}%"

            # Wrinkles (using texture and edge detection)
            wrinkles_area = cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            wrinkles_count = cv2.countNonZero(wrinkles_area)
            conditions["Wrinkles"] = f"{int((wrinkles_count / (w * h)) * 100)}%"

            # Dryness (using skin texture features)
            dryness_area = cv2.inRange(face, (0, 0, 50), (60, 60, 100))
            dryness_count = cv2.countNonZero(dryness_area)
            conditions["Dryness"] = f"{int((dryness_count / (w * h)) * 100)}%"

        return conditions

    def ask_skin_type_questions():
        """Ask the user a series of questions to determine their skin type."""
        print("Please answer the following questions to determine your skin type.")
        print("Type 'yes', 'no', or 'don't know'.")

        questions = {
            "Do you experience shiny or oily areas on your face?": "oily",
            "Do you often feel tightness or dryness on your face?": "dry",
            "Do you have redness or irritation on your skin?": "sensitive",
            "Does your skin have both oily and dry areas?": "combination",
            "Is your skin generally smooth and balanced?": "normal",
        }

        skin_type_score = {key: 0 for key in ["oily", "dry", "sensitive", "combination", "normal"]}

        for question, skin_type in questions.items():
            answer = input(f"{question} (yes/no/don't know): ").strip().lower()

            if answer == "yes":
                skin_type_score[skin_type] += 1

                # Ask lifestyle questions only if main answer is "yes"
                if skin_type == "oily":
                    lifestyle_q1 = input("Do you often eat fried or oily foods? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you frequently sweat or live in a humid environment? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "dry":
                    lifestyle_q1 = input("Do you often skip moisturizing your skin? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you spend a lot of time in air-conditioned or heated rooms? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "sensitive":
                    lifestyle_q1 = input("Does your skin become irritated after using certain skincare products? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you experience skin irritation in extreme weather conditions? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "combination":
                    lifestyle_q1 = input("Do you notice shine or oiliness on your forehead and nose? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you experience dryness on your cheeks but not on your T-zone? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "normal":
                    lifestyle_q1 = input("Do you rarely experience breakouts or dryness? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Is your skin generally smooth and balanced? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

            elif answer == "don't know":
                print("Let's ask some related lifestyle questions to understand better.")

                if skin_type == "oily":
                    lifestyle_q1 = input("Do you often eat fried or oily foods? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you frequently sweat or live in a humid environment? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "dry":
                    lifestyle_q1 = input("Do you often skip moisturizing your skin? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you spend a lot of time in air-conditioned or heated rooms? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "sensitive":
                    lifestyle_q1 = input("Does your skin become irritated after using certain skincare products? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you experience skin irritation in extreme weather conditions? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "combination":
                    lifestyle_q1 = input("Do you notice shine or oiliness on your forehead and nose? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Do you experience dryness on your cheeks but not on your T-zone? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

                elif skin_type == "normal":
                    lifestyle_q1 = input("Do you rarely experience breakouts or dryness? (yes/no): ").strip().lower()
                    lifestyle_q2 = input("Is your skin generally smooth and balanced? (yes/no): ").strip().lower()
                    if lifestyle_q1 == "yes":
                        skin_type_score[skin_type] += 1
                    if lifestyle_q2 == "yes":
                        skin_type_score[skin_type] += 1

            # If user answered "no", no score or follow-up is added

        # Determine skin type based on the highest score
        max_score = max(skin_type_score.values())
        top_types = [k for k, v in skin_type_score.items() if v == max_score]

        if len(top_types) > 1:
            print("Based on your answers, you may have a combination of the following skin types:")
            print(", ".join(t.capitalize() for t in top_types))
            return top_types
        else:
            determined_skin_type = top_types[0]
            print(f"Based on your answers, your skin type is: {determined_skin_type.capitalize()}.")
            return determined_skin_type



    def store_and_recommend_conditions(conditions, skin_type):
        """Store detected conditions and recommend remedies based on severity and skin type."""
        # Clean and standardize column names
        remedies_df.columns = remedies_df.columns.str.strip().str.lower().str.replace(" ", "_")
        print("Standardized column names:", remedies_df.columns.tolist())

        # Check if required columns are present
        required_columns = ['skin_concern', 'skin_type', 'home_remedy', 'chemical_product']
        missing_columns = [col for col in required_columns if col not in remedies_df.columns]
        if missing_columns:
            print(f"Missing columns in the dataset: {missing_columns}")
            return

        # Convert condition percentages to integers for comparison
        high_severity_conditions = {k: int(v.rstrip('%')) for k, v in conditions.items() if v.endswith('%') and int(v.rstrip('%')) > 50}
        low_severity_conditions = {k: int(v.rstrip('%')) for k, v in conditions.items() if v.endswith('%') and int(v.rstrip('%')) <= 50}

        print("\nLow Severity Conditions (No Remedies Needed):")
        for condition, percentage in low_severity_conditions.items():
            print(f"{condition}: {percentage}%")

        print("\nHigh Severity Conditions (Recommendations):")
        for condition, percentage in high_severity_conditions.items():
            print(f"{condition}: {percentage}%")
            
            # Fetch all matching rows for the condition and skin type
            matching_rows = remedies_df[
                (remedies_df["skin_concern"].str.lower() == condition.lower()) &
                (remedies_df["skin_type"].str.lower() == skin_type.lower())
            ]

            if not matching_rows.empty:
                # Gather all recommendations
                home_remedies = matching_rows["home_remedy"].unique()
                chemical_products = matching_rows["chemical_product"].unique()

                print(f"  Home Remedies: {', '.join(home_remedies)}")
                print(f"  Chemical Products: {', '.join(chemical_products)}")
            else:
                print("  No specific recommendations available.")

    def main():
        """Main function for skin analysis."""
        # Ask questions to determine skin type
        skin_type = ask_skin_type_questions()

        cap = cv2.VideoCapture(0)
        conditions = {}
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                if not check_lighting(frame):
                    cv2.putText(frame, "Please move to better lighting!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)

                    if results.detections:
                        conditions = analyze_skin(frame, results.detections)
                        for detection in results.detections:
                            mp_drawing.draw_detection(frame, detection)

                        y_offset = 30
                        for condition, value in conditions.items():
                            cv2.putText(frame, f"{condition}: {value}%", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_offset += 30

                cv2.imshow("Skin Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        # Provide recommendations based on analyzed conditions and user-provided skin type
        if conditions:
            store_and_recommend_conditions(conditions, skin_type)
                                    y_offset = 60
                        for key, value in conditions.items():
                            text = f"{key}: {value}"
                            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y_offset += 25

                cv2.imshow("Skin Analysis", frame)
                if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' key to exit
                    break

        cap.release()
        cv2.destroyAllWindows()

        # If skin_type is a list (multiple), ask user to pick one
        if isinstance(skin_type, list):
            print("\nMultiple skin types detected:")
            for idx, s in enumerate(skin_type, 1):
                print(f"{idx}. {s.capitalize()}")
            choice = input("Please choose the most relevant skin type (enter number): ")
            try:
                skin_type = skin_type[int(choice) - 1]
            except (IndexError, ValueError):
                print("Invalid choice. Defaulting to first option.")
                skin_type = skin_type[0]

        # Store and recommend based on conditions and skin type
        if conditions:
            store_and_recommend_conditions(conditions, skin_type)
        else:
            print("No face detected. Try again.")


    if __name__ == "__main__":
        main()

