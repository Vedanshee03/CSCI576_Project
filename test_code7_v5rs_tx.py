import cv2
import numpy as np
from tqdm import tqdm
import argparse

class SequentialMultiVideoStitcher:
    def __init__(self, frame_step=5, downscale=0.5,
                 orb_features=10000, sift_features=5000, roi_size=1000, H_ext=110):
        self.frame_step = frame_step
        self.downscale  = downscale
        # ORB and SIFT detectors
        self.detector_orb  = cv2.ORB_create(nfeatures=orb_features)
        self.detector_sift = cv2.SIFT_create(nfeatures=sift_features)
        # Matchers
        self.matcher_orb  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.matcher_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        # ROI size for local panorama matching
        self.roi_size = roi_size
        #
        self.H_ext = H_ext

    def sample_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames, idx = [], 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if idx % self.frame_step == 0:
                f = cv2.resize(f, None,
                               fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_AREA)
                frames.append(f)
            idx += 1
        cap.release()
        return frames

    def sample_frames_adaptive(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        idx = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # FAST: Downscale before calculating difference
                gray_curr = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(cv2.resize(prev_frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                
                # Simple difference
                diff = cv2.absdiff(gray_curr, gray_prev)
                
                # Calculate motion percentage
                motion_percentage = np.mean(diff) / 255.0 * 100.0
                
                # print(f"Motion: {motion_percentage:.2f}%")
                
                # Adapt frame step based on motion
                if motion_percentage > 15.0:  # High motion threshold
                    local_step = 1  # Sample more frequently
                    # print("High motion")
                elif motion_percentage > 12.0:  # Medium motion
                    local_step = 2
                    # print("Medium motion")
                elif motion_percentage > 9.0:  # Normal motion
                    local_step = 3
                    # print("Normal motion")
                else:
                    local_step = 5
                    # print("Slow motion")

                if idx % local_step == 0:
                    # Downsample
                    f = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale,
                                interpolation=cv2.INTER_AREA)
                    frames.append(f)
            else:
                # Always include first frame
                f = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale,
                            interpolation=cv2.INTER_AREA)
                frames.append(f)
                
            prev_frame = frame.copy()
            idx += 1
                
        cap.release()
        return frames

    def detect_and_describe_orb(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector_orb.detectAndCompute(gray, None)

    def detect_and_describe_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector_sift.detectAndCompute(gray, None)

    def match_keypoints(self, desA, desB, matcher, ratio=0.75):
        if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
            return []
        try:
            raw = matcher.knnMatch(desA, desB, k=2)
            return [m for m,n in raw if m.distance < ratio * n.distance]
        except:
            return []  # Handle case where knnMatch fails

    def estimate_homography(self, kpA, kpB, matches):
        """Estimate homography with additional inlier ratio for quality assessment"""
        if len(matches) < 4:
            return None, 0.0
        try:
            ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
            ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
            
            # Calculate inlier ratio for quality assessment
            inlier_ratio = np.sum(status) / len(status) if status is not None else 0
            
            return H, inlier_ratio
        except:
            return None, 0.0

    def is_homography_valid(self, H, inlier_ratio, min_inlier_ratio=0.2):
        """
        Check if homography is valid using simple heuristics:
        1. Matrix should not be None
        2. Det(H) should not be close to zero
        3. H should not have extreme values
        4. Inliers ratio should be reasonable
        """
        if H is None:
            return False
            
        # Check determinant (shouldn't be too close to zero)
        det = np.linalg.det(H)
        if abs(det) < 1e-3:
            print("Invalid H: determinant too small")
            return False
            
        # Check for extreme values (transformation too aggressive)
        if np.max(np.abs(H)) > self.H_ext:
            print("Invalid H: values too extreme")
            return False
            
        # Check inlier ratio - at least 40% of matches should be inliers
        if inlier_ratio < min_inlier_ratio:
            print(f"Low quality H: inlier ratio = {inlier_ratio:.2f}")
            return False
            
        return True

    def compute_chain_homographies(self, frames):
        """Compute chain homographies with improved validation and skipping"""
        # Initialize with identity homography for first frame
        Hs = [np.eye(3, dtype=np.float64)]
        H_cum = np.eye(3, dtype=np.float64)
        
        # Get features from first frame with both ORB and SIFT
        kp_prev_orb, des_prev_orb = self.detect_and_describe_orb(frames[0])
        kp_prev_sift, des_prev_sift = self.detect_and_describe_sift(frames[0])
        
        # For tracking bad homographies
        prev_valid_H = np.eye(3, dtype=np.float64)
        skipped_frames = 0
        max_consecutive_skips = 5  # Maximum allowed consecutive skips
        last_good_frame_idx = 0    # Track the last good frame index
        
        for i, img in enumerate(tqdm(frames[1:], desc="Estimating H per frame", leave=False)):
            # Extract features using both ORB and SIFT
            kp_cur_orb, des_cur_orb = self.detect_and_describe_orb(img)
            kp_cur_sift, des_cur_sift = self.detect_and_describe_sift(img)
            
            # Match features using ORB and SIFT
            matches_orb = self.match_keypoints(des_cur_orb, des_prev_orb, self.matcher_orb)
            matches_sift = self.match_keypoints(des_cur_sift, des_prev_sift, self.matcher_sift)
            
            # Combine keypoints and matches
            kp_cur_combined = kp_cur_orb + kp_cur_sift
            kp_prev_combined = kp_prev_orb + kp_prev_sift
            matches_combined = []
            
            # Add ORB matches directly (indices already correct)
            matches_combined.extend(matches_orb)
            
            # Adjust indices for SIFT matches
            offset_cur = len(kp_cur_orb)
            offset_prev = len(kp_prev_orb)
            
            for m in matches_sift:
                adjusted_match = cv2.DMatch()
                adjusted_match.queryIdx = m.queryIdx + offset_cur
                adjusted_match.trainIdx = m.trainIdx + offset_prev
                adjusted_match.distance = m.distance
                matches_combined.append(adjusted_match)
            
            # Calculate homography from combined matches
            H_cur, inlier_ratio = self.estimate_homography(kp_cur_combined, kp_prev_combined, matches_combined)
            
            # Check if homography is valid
            if H_cur is not None and self.is_homography_valid(H_cur, inlier_ratio):
                # Update cumulative homography
                H_cum = H_cum @ H_cur
                
                # Update previous keypoints and descriptors
                kp_prev_orb, des_prev_orb = kp_cur_orb, des_cur_orb
                kp_prev_sift, des_prev_sift = kp_cur_sift, des_cur_sift
                
                # Reset skipped frames counter
                skipped_frames = 0
                prev_valid_H = H_cur
                last_good_frame_idx = i + 1  # Update last good frame index
            else:
                # Bad homography detected
                print(f"Bad homography at frame {i+1}, using fallback strategy")
                
                if skipped_frames < 3:  # Limit consecutive frame skips
                    # Try to use previous valid homography as an estimate
                    H_cum = H_cum @ prev_valid_H
                    skipped_frames += 1
                else:
                    # If too many consecutive failures, just use identity (skip the frame)
                    print(f"Too many consecutive bad homographies, skipping frame {i+1}")
                    # No update to H_cum
                    skipped_frames += 1
                    
                    # If we've skipped too many frames in a row, stop processing this video
                    if skipped_frames >= max_consecutive_skips:
                        print(f"ERROR: Exceeded maximum consecutive skipped frames ({max_consecutive_skips}).")
                        print(f"Stopping chain at frame {i+1}/{len(frames)}.")
                        # Return homographies up to the last good frame (not including bad frames)
                        return Hs[:last_good_frame_idx+1]
            
            # Add the current cumulative homography to the list
            Hs.append(H_cum.copy())
            
        return Hs

    def warp_corners(self, img, H):
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        return warped.reshape(-1, 2)

    def stitch_single_video(self, frames):
        """Create a panorama from a single video's frames with improved canvas sizing and robustness"""
        if not frames:
            return None
        
        # Compute chain homographies with improved method
        Hs_vid = self.compute_chain_homographies(frames)
        
        # Update frames to match the homographies (in case we truncated)
        frames = frames[:len(Hs_vid)]
        
        if len(frames) < 3:
            print("Too few frames after filtering, skipping video")
            return None
        
        # Calculate canvas size for all frames
        pts = [self.warp_corners(frames[i], Hs_vid[i]) for i in range(len(frames))]
        all_pts = np.vstack(pts)
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)

        # Check if canvas size is unreasonably large
        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)
        
        if canvas_width > 6000 or canvas_height > 3000:
            print(f"[WARN] Initial canvas size too large: {canvas_width}x{canvas_height}")
            print("[INFO] Limiting canvas size to maximum dimensions")
            
            # If aspect ratio is maintained, scale to max dimension
            aspect_ratio = canvas_width / max(1, canvas_height)
            if aspect_ratio > 2.0:  # width-dominant
                canvas_width = 6000
                canvas_height = min(3000, int(canvas_width / aspect_ratio))
            else:  # height-dominant or square-ish
                canvas_height = 3000
                canvas_width = min(6000, int(canvas_height * aspect_ratio))
            
            # Recalculate transformation based on new canvas size
            scale_x = canvas_width / (max_x - min_x)
            scale_y = canvas_height / (max_y - min_y)
            scale = min(scale_x, scale_y)
            
            # Apply scaling to transformation
            min_x = min_x * scale
            min_y = min_y * scale
            max_x = min_x + canvas_width
            max_y = min_y + canvas_height

        trans_x, trans_y = -int(min_x), -int(min_y)
        W = int(max_x - min_x)
        H = int(max_y - min_y)

        # Create canvas
        acc_img = np.zeros((H, W, 3), np.float32)
        acc_w = np.zeros((H, W), np.float32)
        H_canvas = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=np.float64)

        # Track consecutive bad frames for early termination
        consecutive_bad_frames = 0
        max_video_consecutive_skips = 10  # Maximum allowed consecutive skips
        last_good_frame_idx = 0
        
        # Accumulate all frames
        for i, frame in enumerate(tqdm(frames, desc="Accumulating frames", leave=False)):
            Hglob = H_canvas @ Hs_vid[i]
            
            # Reset counter for good frames
            consecutive_bad_frames = 0
            last_good_frame_idx = i
            
            # Use a Gaussian weight mask that reduces weight at the edges
            h, w = frame.shape[:2]
            y, x = np.mgrid[0:h, 0:w]
            center_y, center_x = h // 2, w // 2
            weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (max(w, h)**2))
            
            # Keep weight as single channel
            weight = weight.astype(np.float32)
            
            # Warp frame and weight
            warped = cv2.warpPerspective(frame.astype(np.float32), Hglob, dsize=(W, H))
            warped_weight = cv2.warpPerspective(weight, Hglob, dsize=(W, H))
            
            # Add to accumulation - ensure correct broadcasting
            mask = (warped.max(axis=2) > 0).astype(np.float32)
            
            # Apply weights - expand where needed for broadcasting with 3-channel warped
            acc_img += warped * warped_weight[:, :, np.newaxis]
            acc_w += warped_weight * mask  # Both are single channel
        
        # Normalize - avoid division by zero
        acc_w[acc_w == 0] = 1.0
        
        # Broadcast acc_w to match acc_img shape for division
        pano = (acc_img / acc_w[:, :, np.newaxis]).astype(np.uint8)
        
        # Crop to valid region
        mask = (acc_w > 0)
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return None
        pano = pano[ys.min():ys.max()+1, xs.min():xs.max()+1]
        
        return pano

    def stitch_panoramas(self, panoramas):
        """Stitch multiple panoramas together using OpenCV's Stitcher"""
        if not panoramas:
            return None
        if len(panoramas) == 1:
            return panoramas[0]
        
        print(f"Stitching {len(panoramas)} panoramas together using OpenCV Stitcher")
        
        try:
            # Create an OpenCV Stitcher object
            # Use cv2.Stitcher.create() for OpenCV 3.x or higher
            # Use cv2.createStitcher() for OpenCV 2.x
            try:
                stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            except:
                # Fallback for older OpenCV versions
                stitcher = cv2.createStitcher(False)
            
            # Attempt to stitch the panoramas
            status, stitched = stitcher.stitch(panoramas)
            
            # Check if stitching was successful
            if status == cv2.Stitcher_OK:
                print("Panorama stitching successful!")
                return stitched
            else:
                print(f"Panorama stitching failed with error code: {status}")
                if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                    print("Error: Need more images")
                elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
                    print("Error: Homography estimation failed")
                elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
                    print("Error: Camera parameter adjustment failed")
                
                print("Falling back to side-by-side placement")
                # Simple side-by-side placement as a fallback
                height = max(p.shape[0] for p in panoramas)
                width = sum(p.shape[1] for p in panoramas)
                result = np.zeros((height, width, 3), dtype=np.uint8)
                
                current_width = 0
                for p in panoramas:
                    h, w = p.shape[:2]
                    result[:h, current_width:current_width+w] = p
                    current_width += w
                
                return result
                
        except Exception as e:
            print(f"Error in panorama stitching: {e}")
            
            # Simple side-by-side placement as a fallback
            height = max(p.shape[0] for p in panoramas)
            width = sum(p.shape[1] for p in panoramas)
            result = np.zeros((height, width, 3), dtype=np.uint8)
            
            current_width = 0
            for p in panoramas:
                h, w = p.shape[:2]
                result[:h, current_width:current_width+w] = p
                current_width += w
            
            return result

    def stitch_multi_videos(self, video_paths):
        """Main function to stitch multiple videos by first stitching each video separately"""
        # First, create individual panoramas for each video
        video_panos = []
        
        for vid_idx, path in enumerate(tqdm(video_paths, desc="Processing individual videos")):
            print(f"Processing video {vid_idx+1}/{len(video_paths)}: {path}")
            try:
                frames = self.sample_frames_adaptive(path)
                if not frames:
                    print(f"[WARN] no frames in {path}, skipping")
                    continue
                    
                # Create panorama for this single video
                vid_pano = self.stitch_single_video(frames)
                if vid_pano is not None:
                    # Save intermediate panorama (optional)
                    cv2.imwrite(f"pano_vid_{vid_idx}.jpg", vid_pano)
                    video_panos.append(vid_pano)
            except Exception as e:
                print(f"Error processing video {path}: {e}")
                
        # Now stitch together the individual panoramas
        if not video_panos:
            print("No valid panoramas created from videos")
            return None
        elif len(video_panos) == 1:
            print("Only one valid panorama created")
            return video_panos[0]
        else:
            return self.stitch_panoramas(video_panos)

    def stitch(self, video_paths):
        """Main entry point for stitching - calls stitch_multi_videos"""
        return self.stitch_multi_videos(video_paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+",
                    help="Input videos, in the order to stitch")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--down", type=float, default=0.5,
                   help="Downscale factor (0<down<=1)")
    p.add_argument("--output", default="panorama.jpg")
    args = p.parse_args()

    stitcher = SequentialMultiVideoStitcher(
        frame_step=args.step,
        downscale=args.down,
    )
    pano = stitcher.stitch(args.videos)
    if pano is not None:
        cv2.imwrite(args.output, pano)
        print("Saved:", args.output)
    else:
        print("Failed to create panorama")

if __name__ == "__main__":
    main()