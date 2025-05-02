import cv2
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing as mp
import time

class VideoPanoramaStitcher:
    def __init__(self, frame_step=5, downscale=0.5, batch_size=10):
        self.frame_step = frame_step
        self.downscale = downscale
        self.batch_size = batch_size  # Number of frames to process in a batch


    def sample_frames_in_batches(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {path}")

        batch = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                # end of video: if there's a partial batch, emit it
                if batch:
                    yield batch
                break

            # skip frames according to frame_step
            if idx % self.frame_step != 0:
                idx += 1
                continue

            # downscale
            frame = cv2.resize(
                frame,
                None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )

            batch.append(frame)
            # once we have batch_size frames, emit and reset
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            idx += 1

        cap.release()

    def detect_and_describe_orb(self, img):
        # Create ORB detector inside the method
        detector_orb = cv2.ORB_create(20000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return detector_orb.detectAndCompute(gray, None)

    def detect_and_describe_sift(self, img):
        # Create SIFT detector inside the method
        detector_sift = cv2.SIFT_create(20000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return detector_sift.detectAndCompute(gray, None)

    def match_keypoints_orb(self, A, B, ratio=0.75):
        raw = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(A, B, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def match_keypoints_sift(self, A, B, ratio=0.75):
        raw = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(A, B, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def estimate_homography(self, kpA, kpB, matches):
        if len(matches) < 4:
            return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
        return H

    def stitch_batch(self, batch):
        """Stitch a batch of frames together"""
        pano = batch[0].copy()
        kp_p_orb, des_p_orb = self.detect_and_describe_orb(pano)
        kp_p_sift, des_p_sift = self.detect_and_describe_sift(pano)

        for img in tqdm(batch[1:], desc="Stitching frames in batch"):
            kp_i_orb, des_i_orb = self.detect_and_describe_orb(img)
            kp_i_sift, des_i_sift = self.detect_and_describe_sift(img)

            # match ORB
            if des_i_orb is None or des_p_orb is None:
                matches_orb = []
            else:
                matches_orb = self.match_keypoints_orb(des_i_orb, des_p_orb)

            # match SIFT
            if des_i_sift is None or des_p_sift is None:
                matches_sift = []
            else:
                matches_sift = self.match_keypoints_sift(des_i_sift, des_p_sift)

            # Combine keypoints from ORB and SIFT
            kp_i_combined = kp_i_orb + kp_i_sift
            kp_p_combined = kp_p_orb + kp_p_sift

            # Combine matches from ORB and SIFT
            matches_combined = matches_orb + matches_sift
            if len(matches_combined) < 4:
                print("Not enough matches found.")
                continue

            # Estimate homography using the combined keypoints and matches
            H = self.estimate_homography(kp_i_combined, kp_p_combined, matches_combined)
            if H is None:
                continue

            # compute canvas
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            corners = np.float32([[0, 0], [0, h_i], [w_i, h_i], [w_i, 0]]).reshape(-1, 1, 2)
            warped_c = cv2.perspectiveTransform(corners, H)
            all_c = np.vstack((np.float32([[0, 0], [0, h_p], [w_p, h_p], [w_p, 0]]).reshape(-1, 1, 2), warped_c))
            x0, y0 = np.int32(all_c.min(axis=0).ravel() - 0.5)
            x1, y1 = np.int32(all_c.max(axis=0).ravel() + 0.5)
            trans = [-x0, -y0]
            Ht = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]], dtype=np.float64)
            canvas = (x1 - x0, y1 - y0)

            # warp both into canvas
            pano_warped = cv2.warpPerspective(pano, Ht, canvas)
            img_w = cv2.warpPerspective(img, Ht.dot(H), canvas)

            # Feather blend
            mask_p = (pano_warped[:, :, 0] > 0).astype(np.uint8)
            mask_i = (img_w[:, :, 0] > 0).astype(np.uint8)

            # distance transform on each mask
            dp = cv2.distanceTransform(mask_p, cv2.DIST_L2, 5).astype(np.float32)
            di = cv2.distanceTransform(mask_i, cv2.DIST_L2, 5).astype(np.float32)
            wsum = dp + di + 1e-6  # avoid division by zero

            # normalized weights
            wp = (dp / wsum)[:, :, None]
            wi = (di / wsum)[:, :, None]

            # blend and convert back to uint8
            pano = (pano_warped.astype(np.float32) * wp + img_w.astype(np.float32) * wi).astype(np.uint8)

            # update features
            kp_p_orb, des_p_orb = self.detect_and_describe_orb(pano)
            kp_p_sift, des_p_sift = self.detect_and_describe_sift(pano)

        return pano

    def stitch(self, videos):
        """Stitch multiple videos into a panorama"""
        start_time = time.time()

        # 1) Sample & downscale frames in batches
        all_batches = []
        for v in videos:
            for batch in self.sample_frames_in_batches(v):
                all_batches.append(batch)

        print(f"→ Will process {len(all_batches)} batch(es) of up to {self.batch_size} frames each")
        if not all_batches:
            raise RuntimeError("No frames extracted")
        
        # 2) Parallel stitch each batch
        with mp.Pool(mp.cpu_count()) as pool:
            stitched_batches = pool.map(self.stitch_batch, all_batches)

        #3) Merge batch-pano results together
        final_pano = stitched_batches[0]
        for pano in stitched_batches[1:]:
            final_pano = self.merge_panorama(final_pano, pano)

        elapsed = time.time() - start_time
        print(f"Total stitching time: {elapsed:.2f} seconds")
        return final_pano

    def merge_panorama(self, pano1, pano2):
        #1) feature detect 
        kp1, des1 = self.detect_and_describe_orb(pano1)
        kp2, des2 = self.detect_and_describe_orb(pano2)

        #2) match and filter 
        matches = self.match_keypoints_orb(des2, des1)
        if len(matches) < 4:
            # fallback to simple hstack if not enough matches
            return np.hstack((pano1, pano2))
        
        # 3) homography
        H = self.estimate_homography(kp2, kp1, matches)
        if H is None:
            return np.hstack((pano1, pano2))
        
        # 4) compute merged canvas size
        h1, w1 = pano1.shape[:2]
        h2, w2 = pano2.shape[:2]
        corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        warped_c2 = cv2.perspectiveTransform(corners2, H)
        all_c = np.vstack([
            np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2),
            warped_c2
        ])
        x0,y0 = np.int32(all_c.min(axis=0).ravel() - 0.5)
        x1,y1 = np.int32(all_c.max(axis=0).ravel() + 0.5)
        trans = [-x0, -y0]
        Ht = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]],dtype=np.float64)
        canvas_size = (x1-x0, y1-y0)

        # 5) warp & blend
        base = cv2.warpPerspective(pano1, Ht, canvas_size)
        over = cv2.warpPerspective(pano2, Ht.dot(H), canvas_size)

        mask_b = (base>0).any(axis=2).astype(np.uint8)
        mask_o = (over>0).any(axis=2).astype(np.uint8)
        db = cv2.distanceTransform(mask_b,cv2.DIST_L2,5).astype(np.float32)
        do = cv2.distanceTransform(mask_o,cv2.DIST_L2,5).astype(np.float32)
        wsum = db+do+1e-6
        wb = (db/wsum)[:,:,None]
        wo = (do/wsum)[:,:,None]

        merged = (base.astype(np.float32)*wb + over.astype(np.float32)*wo).astype(np.uint8)
        return merged


def main():
    parser = argparse.ArgumentParser(description="Advanced Video Panorama Stitcher")
    parser.add_argument("videos", nargs="+", help="Input video files")
    parser.add_argument(
        "--output",
        "-o",
        default="panorama.jpg",
        help="Output image path",
    )

    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=5,
        help="Frame step (number of frames to skip)",
    )

    parser.add_argument(
        "--down",
        type=float,
        default=0.5,
        help="Downscale factor (0-1)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10, #changed to batch size 30 so it is faster 
        help="Number of frames to process in a batch",
    )

    args = parser.parse_args()

    stitcher = VideoPanoramaStitcher(
        frame_step=args.step,
        downscale=args.down,
        batch_size=args.batch_size,
    )
    pano = stitcher.stitch(args.videos)
    cv2.imwrite(args.output, pano)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
