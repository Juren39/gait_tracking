# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from pathlib import Path
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from opengait.Gait_Model import GaitModel
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.utils.association import associate, linear_assignment
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.ops import xyxy2xysr

from dataset.data_store import check_and_record, get_track_id_by_number

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, det, delta_t=3, emb=None, alpha=0, max_obs=100, Q_xy_scaling = 0.01, Q_s_scaling = 0.0001):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.max_obs=max_obs
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                # x  y  s  r  x' y' s'
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.bbox_to_z_func = xyxy2xysr
        self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.features = deque([], maxlen=self.max_obs)
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.history_observations = deque([], maxlen=self.max_obs)
        self.trajectory_buffer = deque([], maxlen=self.max_obs)  # 存储 30 帧轨迹
        self.gait_feature = None
        self.emb = emb

        self.frozen = False


    def update(self, det):
        """
        Updates the state vector with observed bbox.
        """
        if det is not None:
            bbox = det[0:5]
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(det)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
    
    def update_trajectory(self, img, det):
        """
        更新历史步态轨迹，具体存储形式为检测框图片。
        """
        bbox = det[0:4]
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox.astype('int')
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        self.trajectory_buffer.append(img[y1:y2, x1:x2])

    def should_match_gait(self):
        """
        攒够足够长的长度，需要推进去得到轨迹的步态特征
        """
        return len(self.trajectory_buffer) >= self.max_obs

class DeepOcSort(BaseTracker):
    """
    DeepOCSort Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking.

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        det_thresh (float, optional): Detection confidence threshold. Detections below this threshold will be ignored.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        min_hits (int, optional): Minimum number of hits required to confirm a track.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for data association.
        delta_t (int, optional): Time delta for velocity estimation in Kalman Filter.
        asso_func (str, optional): Association function to use for data association. Options include "iou" for IoU-based association.
        inertia (float, optional): Weight for inertia in motion modeling. Higher values make tracks less responsive to changes.
        w_association_emb (float, optional): Weight for the embedding-based association score.
        alpha_fixed_emb (float, optional): Fixed alpha for updating embeddings. Controls the contribution of new and old embeddings in the ReID model.
        aw_param (float, optional): Parameter for adaptive weighting between association costs.
        embedding_off (bool, optional): Whether to turn off the embedding-based association.
        cmc_off (bool, optional): Whether to turn off camera motion compensation (CMC).
        aw_off (bool, optional): Whether to turn off adaptive weighting.
        Q_xy_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for position coordinates.
        Q_s_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for scale coordinates.
        **kwargs: Additional arguments for future extensions or parameters.
    """
    def __init__(
        self,
        reid_weights: Path,
        gait_weights: Path,
        device: torch.device,
        half: bool,
        save_file: dict,
        mode: str,
        per_class: bool = False,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        w_association_emb: float = 0.5,
        alpha_fixed_emb: float = 0.95,
        aw_param: float = 0.5,  
        embedding_off: bool = False,
        cmc_off: bool = False,
        aw_off: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        **kwargs: dict
    ):
        super().__init__(max_age=max_age, per_class=per_class, asso_func=asso_func)
        """
        Sets key parameters for SORT
        """
        self.save_file = save_file
        self.mode = mode
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = asso_func
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        KalmanBoxTracker.count = 1

        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model
        self.gait_model = GaitModel(
            device,
            cfgs_path=gait_weights
        )
        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = get_cmc_method('sof')()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score,cls],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # dets, s, c = dets.data
        # print(dets, s, c)
        self.check_inputs(dets, img)

        self.frame_count += 1
        self.height, self.width = img.shape[:2]
        # 对检测框 dets 进行预处理，筛选出置信度大于 det_thresh（默认 0.3）的目标。
        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)]) # 给 dets 追加索引编号，便于跟踪数据关联
        assert dets.shape[1] == 7
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        # dets -> [x1, y1, x2, y2, score, class_id, det_index]

        # appearance descriptor extraction
        if self.embedding_off or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        elif embs is not None:
            dets_embs = embs
        else:
            dets_embs = self.model.get_features(dets[:, 0:4], img) # Re-id外观特征提取 -> dets_embs
        # CMC处理，全局运动补偿，消除摄像机运动对目标跟踪的影响。
        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh) #计算并归一化trust
        af = self.alpha_fixed_emb
        # self.alpha_fixed_emb 是 固定的 alpha 值，用于 特征更新的平滑权重，类似于 指数平滑。
        # 值越大，表示 新特征对旧特征的影响较小，平滑更新。值越小，表示 新特征更新得更快，适应性更强。

        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)
        # 计算 dets_alpha，用于更新 ReID 特征
        # trust 越高，说明检测器置信度越高，1 - trust 就越小，最终 dets_alpha 越接近 self.alpha_fixed_emb，表示 保留更多旧特征。
        # trust 越低，说明检测器置信度较低，1 - trust 越大，最终 dets_alpha 趋向 1，表示 更快更新新特征。

        # get predicted locations from existing trackers.
        # 预测当前活跃轨迹 self.active_tracks 的下一时刻位置，并获取其外观特征 trk_embs
        trks = np.zeros((len(self.active_tracks), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0] # 通过 卡尔曼滤波器 预测下一时刻的目标位置
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): # 如果 pos 包含 NaN 值（可能是由于预测失败或初始化问题）
                to_del.append(t)
            else:
                trk_embs.append(self.active_tracks[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) 

        if len(trk_embs) > 0: # trk_embs 存储的是 所有活跃轨迹的 ReID 特征向量，用于目标关联匹配。
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t) # 清理无效轨迹
        
        #计算运动信息， 最后一次观测的边界框， 历史观测点
        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.active_tracks])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])

        """
            First round of association
        """
        ####################################
        #       compute stage1_emb_cost
        ####################################
        # (M detections X N tracks, final score
        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.asso_func,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            img.shape[1], # w
            img.shape[0], # h
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )
        for m in matched:
            trk = self.active_tracks[m[1]]
            trk.update(dets[m[0], :])
            trk.update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
            if self.mode == "recognition":
                trk.update_trajectory(img, dets[m[0], :])
                if trk.should_match_gait() and trk.gait_feature is None:
                    trk.gait_feature = self.gait_model.extract_gait_feature(
                        trk.trajectory_buffer)
                    trk.id = check_and_record(
                        trk.emb, trk.gait_feature, self.mode, self.save_file, trk.id, threshold=0.65) 
                    print(trk.id)

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            # TODO: is better without this
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    trk = self.active_tracks[trk_ind]
                    trk.update(dets[det_ind, :])
                    trk.update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    if self.mode == "recognition":
                        trk.update_trajectory(img, dets[m[0], :])
                        if trk.should_match_gait() and trk.gait_feature is None:
                            trk.gait_feature = self.gait_model.extract_gait_feature(
                                trk.trajectory_buffer)
                            trk.id = check_and_record(
                                trk.emb, trk.gait_feature, self.mode, self.save_file, trk.id, threshold=0.65) 
                            print(trk.id)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                Q_xy_scaling=self.Q_xy_scaling, 
                Q_s_scaling=self.Q_s_scaling,                
                max_obs=self.max_obs
            )
            if self.mode == "recognition":
                trk.update_trajectory(img, dets[i, :])
                trk.id = 0
            self.active_tracks.append(trk)
            
        i = len(self.active_tracks)

        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id], [trk.conf], [trk.cls], [trk.det_ind] )).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                if self.mode == "recognition" and trk.gait_feature is None:
                    trk.gait_feature = self.gait_model.extract_gait_feature(
                        trk.trajectory_buffer)
                    trk.id = check_and_record(
                        trk.emb, trk.gait_feature, self.mode, self.save_file, trk.id, threshold=0.65) 
                    print(trk.id)
                self.active_tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])
    
    def registration(self, dets: list, imgs: list = None) -> list: 

        if len(dets) == 0:
            return
        assert len(imgs) == len(dets), "imgs 和 dets 的帧数不匹配"
        for f_idx, (frame, frame_dets) in enumerate(zip(imgs, dets)):
            if frame_dets is None or len(frame_dets) == 0:
                continue
            dets_embs = self.model.get_features(frame_dets[:, 0:4], frame)
            trust = (frame_dets[:, 4] - self.det_thresh) / (1 - self.det_thresh) #计算并归一化trust
            af = self.alpha_fixed_emb
            dets_alpha = af + (1 - af) * (1 - trust)
            
            # 裁剪并记录ID
            for i, det in enumerate(frame_dets):
                x1, y1, x2, y2, score = det[:5].astype(int) # [x1, y1, x2, y2, confidence, cls, det_id]
                det_id = str(get_track_id_by_number(Path('./dataset/id_name_mapping.txt'), det[6]))
                match = False

                for trk in self.active_tracks:
                    if det_id == trk.id:
                        trk.update(frame_dets[i, :])
                        trk.update_emb(dets_embs[i], alpha=dets_alpha[i])
                        trk.update_trajectory(frame, frame_dets[i, :])
                        match = True
                        break
                if not match:
                    trk = KalmanBoxTracker(
                        frame_dets[i],
                        delta_t=self.delta_t,
                        emb=dets_embs[i],
                        alpha=dets_alpha[i],
                        Q_xy_scaling=self.Q_xy_scaling, 
                        Q_s_scaling=self.Q_s_scaling,                
                        max_obs=self.max_obs,
                    )
                    trk.id = det_id
                    trk.trajectory_buffer = deque([], maxlen=len(dets))
                    trk.update_trajectory(frame, frame_dets[i, :])
                    self.active_tracks.append(trk)
        box_id_list = []
        for trk in self.active_tracks:
            trk.gait_feature = self.gait_model.extract_gait_feature(trk.trajectory_buffer)
            matched_id = check_and_record(trk.emb, trk.gait_feature, self.mode, self.save_file, trk.id, threshold=0.65) 
            box_id_list.append(trk.id)
        return box_id_list
    
    def recognition(self, dets: list, imgs: list = None):

        dets_info_list = []
        dets_info = self.update(dets, imgs)
        for det in dets_info:
            if det[1][0] != 0 and det[1][0] not in self.register_tracks:
                self.register_tracks.append(det[1][0])
                dets_info_list.append(self.active_tracks[det[4][0]].register_tracks)
        return dets_info, dets_info_list




