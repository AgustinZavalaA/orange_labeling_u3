import pickle
import cv2
import numpy as np
import pandas as pd
from frame_labeler import frame_labeler


def process_slit_frame(frame: np.array, size: int) -> np.array:
    frame = cv2.resize(frame, (size, size))
    axis = frame.shape[1] // 2

    return frame[:, axis]


def get_slit_image(video_path: str, size: int) -> np.array:
    cap = cv2.VideoCapture(video_path)

    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    slit_image = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), size, 3), np.uint8)
    n_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        slit_image[n_frames] = process_slit_frame(frame, size)
        n_frames += 1

    slit_image = cv2.transpose(slit_image)
    return slit_image


def get_contours_from_slit_image(slit_image: np.array) -> np.array:
    gray_image = cv2.cvtColor(slit_image, cv2.COLOR_BGR2GRAY)
    bin_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(bin_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 60]

    return filtered_contours


def get_centroids_from_contours(contours: np.array) -> np.array:
    centroids = np.zeros((len(contours), 2), np.float32)
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        centroids[i] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return centroids


def label_video_frames(video_path: str, centroids: np.array) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    f_labeler = None
    n_frames = 0
    labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        n_frames += 1
        if n_frames in centroids[:, 0]:
            cv2.imshow(f"Frame {n_frames}", frame)
            f_labeler = frame_labeler(str(n_frames))

            # cv2.waitKey(0)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                f_labeler = None
                return labels

            labels.append(f"{n_frames},{f_labeler.get_label()}")
            # print(labels[-1])
            cv2.destroyAllWindows()


def write_labels_to_csv(labels: list[str], video_path: str) -> None:
    df = pd.DataFrame([label.split(",") for label in labels], columns=["frame", "size", "state"])
    # print(df)
    df.to_csv(f"{video_path.split('.')[0]}_labels.csv", index=False)


def main(process_video: bool = True, video_path: str = "dataset/lento_1.MOV", size: int = 256) -> None:
    if process_video:
        slit_image = get_slit_image(video_path, size)
        # save slit image
        cv2.imwrite(f"{video_path.split('.')[0]}.png", slit_image)
    else:
        slit_image = cv2.imread(f"{video_path.split('.')[0]}.png")

    contours = get_contours_from_slit_image(slit_image)
    print(f"Found {len(contours)} contours")

    contours_image = cv2.drawContours(slit_image.copy(), contours, -1, (0, 255, 0), 2)
    # cv2.imwrite(f"{video_path.split('.')[0]}_contours.png", contours_image)

    cv2.imshow("Slit Image", slit_image)
    cv2.imshow("Contours Image", contours_image)

    centroids = get_centroids_from_contours(contours)

    labels = label_video_frames(video_path, centroids)
    print(labels)

    write_labels_to_csv(labels, video_path)

    cv2.waitKey(0)


if __name__ == "__main__":
    main(
        process_video=False,
        video_path="dataset/lento_1.MOV",
    )
