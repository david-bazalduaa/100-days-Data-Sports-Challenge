"""
Day 17 & Day 18: Feature Engineering for Expected Goals (xG) Model

This module provides modular, reusable, and production-grade feature engineering
functions for football (soccer) event data based on the StatsBomb coordinate system (120x80).

Feature Categories:
1. Geometric Features:
   - Euclidean distance to goal center (120, 40)
   - Angle between goal posts (120, 36) and (120, 44)
   - Distances to left and right posts
   - Distance to nearest post & post distance asymmetry ratio
   - Lateral shooting channel zoning (3-zone and 5-zone positional play)
   - Visible goal angle discounting goalkeeper occlusion
2. Contextual & Technical Features:
   - Body part (right foot, left foot, head, other)
   - Play situation (open play, set piece, counter attack, penalty)
   - Assistance type from preceding key pass (cross, through ball, cutback, recovery, set piece, unassisted)
   - Context flags (first time shot, follows dribble, open goal, one on one, under pressure)
3. Defensive Pressure & Freeze Frame Features:
   - Opponents inside the triangular shot cone (defenders_in_cone)
   - Distance to nearest outfield defender
   - Goalkeeper coordinates (goalkeeper_x, goalkeeper_y)
   - Goalkeeper distance to goal, depth from goal line, and perpendicular line deviation
4. Possession Sequence Features:
   - Possession sequence duration (seconds)
   - Number of passes in possession leading to shot
   - Total events in sequence and pitch progression (start_x to shot_x)
5. Temporal & Scoreline State Features:
   - Exact game minute
   - Pre-shot match score differential and categorical game state (leading, drawing, trailing)
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import logging

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Pitch Dimensions & Goal Constants (StatsBomb Coordinate System: 120 x 80 yards)
PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
GOAL_CENTER_X: float = 120.0
GOAL_CENTER_Y: float = 40.0
POST_LEFT: Tuple[float, float] = (120.0, 36.0)
POST_RIGHT: Tuple[float, float] = (120.0, 44.0)


# ==============================================================================
# 1. GEOMETRIC FEATURES
# ==============================================================================

def extract_shot_coordinates(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts numerical X and Y shot coordinates from a DataFrame.
    Handles both array-like 'location' columns ([x, y]) and separated
    'location_x' / 'location_y' columns.

    Args:
        df: DataFrame containing shot events.

    Returns:
        Tuple of (shot_x, shot_y) as float Series.
    """
    if "location_x" in df.columns and "location_y" in df.columns:
        shot_x = pd.to_numeric(df["location_x"], errors="coerce")
        shot_y = pd.to_numeric(df["location_y"], errors="coerce")
        if shot_x.notna().sum() > 0:
            return shot_x, shot_y

    if "location" in df.columns:
        shot_x = df["location"].apply(
            lambda loc: float(loc[0]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
        )
        shot_y = df["location"].apply(
            lambda loc: float(loc[1]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
        )
        return shot_x, shot_y

    raise KeyError("Could not find valid 'location' or 'location_x'/'location_y' in DataFrame.")


def calculate_shot_distance(
    shot_x: np.ndarray,
    shot_y: np.ndarray,
    goal_x: float = GOAL_CENTER_X,
    goal_y: float = GOAL_CENTER_Y,
) -> np.ndarray:
    """
    Calculates the Euclidean distance in yards from the shot location to the goal center.
    """
    dx = goal_x - np.asarray(shot_x, dtype=float)
    dy = goal_y - np.asarray(shot_y, dtype=float)
    return np.sqrt(dx**2 + dy**2)


def calculate_shot_angle(
    shot_x: np.ndarray,
    shot_y: np.ndarray,
    post_left: Tuple[float, float] = POST_LEFT,
    post_right: Tuple[float, float] = POST_RIGHT,
    in_degrees: bool = False,
) -> np.ndarray:
    """
    Calculates the visible goal angle (subtended angle between the two goal posts)
    from the shot position.
    """
    sx = np.asarray(shot_x, dtype=float)
    sy = np.asarray(shot_y, dtype=float)

    angle_right = np.arctan2(post_right[1] - sy, post_right[0] - sx)
    angle_left = np.arctan2(post_left[1] - sy, post_left[0] - sx)

    shot_angle = np.abs(angle_right - angle_left)
    shot_angle = np.where(shot_angle > np.pi, 2 * np.pi - shot_angle, shot_angle)

    if in_degrees:
        return np.degrees(shot_angle)
    return shot_angle


def calculate_nearest_post_features(
    dist_left: np.ndarray,
    dist_right: np.ndarray,
) -> pd.DataFrame:
    """
    Calculates the Euclidean distance to the nearest goalpost and post asymmetry ratio.
    """
    dl = np.asarray(dist_left, dtype=float)
    dr = np.asarray(dist_right, dtype=float)

    nearest = np.minimum(dl, dr)
    furthest = np.maximum(dl, dr)

    post_ratio = np.divide(
        nearest,
        furthest,
        out=np.zeros_like(nearest, dtype=float),
        where=furthest != 0
    )

    return pd.DataFrame({
        "distance_to_nearest_post": nearest,
        "post_distance_ratio": post_ratio
    })


def calculate_shooting_channel_features(shot_y: np.ndarray) -> pd.DataFrame:
    """
    Zonifies shot locations into lateral channels using StatsBomb pitch width (Y in [0, 80]).
    """
    sy = pd.Series(shot_y)

    # 1. 3-Zone Classification (20 - 40 - 20)
    bins_3zone = [-1.0, 20.0, 60.0, 81.0]
    labels_3zone = ["left_flank", "central_box", "right_flank"]
    channel_3zone = pd.cut(sy, bins=bins_3zone, labels=labels_3zone)

    # 2. 5-Zone Tactical Classification (Positional Play)
    bins_5zone = [-1.0, 18.0, 30.0, 50.0, 62.0, 81.0]
    labels_5zone = ["wide_left", "half_space_left", "central_corridor", "half_space_right", "wide_right"]
    channel_5zone = pd.cut(sy, bins=bins_5zone, labels=labels_5zone)

    # 3. Binary central indicator (Y in [30, 50])
    is_central = ((sy >= 30.0) & (sy <= 50.0)).astype(int)

    return pd.DataFrame({
        "shooting_channel_3zone": channel_3zone.values,
        "shooting_channel_5zone": channel_5zone.values,
        "is_central_shot": is_central.values
    })


def calculate_visible_goal_angle(
    shot_angle_rad: np.ndarray,
    dist_to_gk: np.ndarray,
    gk_in_cone: np.ndarray,
    gk_effective_width: float = 1.8
) -> pd.DataFrame:
    """
    Calculates open goal angle by subtracting the goalkeeper's subtended angle.
    """
    sa = np.asarray(shot_angle_rad, dtype=float)
    dg = np.asarray(dist_to_gk, dtype=float)
    in_cone = np.asarray(gk_in_cone, dtype=float)

    safe_gk_dist = np.maximum(dg, 0.5)
    gk_subtended_angle = 2.0 * np.arctan((gk_effective_width / 2.0) / safe_gk_dist)

    effective_occlusion = np.where(in_cone == 1, gk_subtended_angle, 0.0)
    visible_angle_rad = np.maximum(0.0, sa - effective_occlusion)
    visible_angle_deg = np.degrees(visible_angle_rad)

    visible_ratio = np.divide(
        visible_angle_rad,
        np.maximum(sa, 1e-5),
        out=np.ones_like(visible_angle_rad, dtype=float),
        where=sa > 1e-5
    )
    visible_ratio = np.clip(visible_ratio, 0.0, 1.0)

    return pd.DataFrame({
        "visible_angle_to_goal": visible_angle_rad,
        "visible_angle_to_goal_deg": visible_angle_deg,
        "visible_angle_ratio": visible_ratio
    })


def calculate_geometric_features(
    shot_x: np.ndarray,
    shot_y: np.ndarray,
) -> pd.DataFrame:
    """
    Computes baseline geometric features for shots.
    """
    sx = np.asarray(shot_x, dtype=float)
    sy = np.asarray(shot_y, dtype=float)

    dist_to_center = calculate_shot_distance(sx, sy, GOAL_CENTER_X, GOAL_CENTER_Y)
    dist_to_left_post = calculate_shot_distance(sx, sy, POST_LEFT[0], POST_LEFT[1])
    dist_to_right_post = calculate_shot_distance(sx, sy, POST_RIGHT[0], POST_RIGHT[1])

    angle_rad = calculate_shot_angle(sx, sy, POST_LEFT, POST_RIGHT, in_degrees=False)
    angle_deg = np.degrees(angle_rad)
    eccentricity = np.abs(sy - GOAL_CENTER_Y)

    # Nearest post features
    post_feats = calculate_nearest_post_features(dist_to_left_post, dist_to_right_post)
    channel_feats = calculate_shooting_channel_features(sy)

    return pd.DataFrame({
        "shot_x": sx,
        "shot_y": sy,
        "distance_to_goal": dist_to_center,
        "distance_to_left_post": dist_to_left_post,
        "distance_to_right_post": dist_to_right_post,
        "distance_to_nearest_post": post_feats["distance_to_nearest_post"].values,
        "post_distance_ratio": post_feats["post_distance_ratio"].values,
        "shot_angle": angle_rad,
        "shot_angle_deg": angle_deg,
        "shot_eccentricity": eccentricity,
        "shooting_channel_3zone": channel_feats["shooting_channel_3zone"].values,
        "shooting_channel_5zone": channel_feats["shooting_channel_5zone"].values,
        "is_central_shot": channel_feats["is_central_shot"].values,
    })


# ==============================================================================
# 2. CONTEXTUAL & TECHNICAL FEATURES
# ==============================================================================

def extract_body_part_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes and encodes shot body part features.
    """
    raw_bp = df["shot_body_part"].astype(str).str.lower() if "shot_body_part" in df.columns else pd.Series(["unknown"] * len(df))

    standardized_bp = pd.Series("other", index=df.index)
    standardized_bp[raw_bp.str.contains("right", na=False)] = "right_foot"
    standardized_bp[raw_bp.str.contains("left", na=False)] = "left_foot"
    standardized_bp[raw_bp.str.contains("head", na=False)] = "head"

    return pd.DataFrame({
        "body_part": standardized_bp,
        "is_header": (standardized_bp == "head").astype(int),
        "is_foot": standardized_bp.isin(["right_foot", "left_foot"]).astype(int),
    }, index=df.index)


def extract_play_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the play situation context for each shot.
    """
    shot_type = df["shot_type"].astype(str) if "shot_type" in df.columns else pd.Series([""] * len(df), index=df.index)
    play_pattern = df["play_pattern"].astype(str) if "play_pattern" in df.columns else pd.Series([""] * len(df), index=df.index)

    situation = pd.Series("open_play", index=df.index)

    is_penalty = shot_type.str.contains("Penalty", case=False, na=False)
    is_counter = play_pattern.str.contains("Counter", case=False, na=False)
    is_corner = play_pattern.str.contains("Corner", case=False, na=False) | shot_type.str.contains("Corner", case=False, na=False)
    is_fk = play_pattern.str.contains("Free Kick", case=False, na=False) | shot_type.str.contains("Free Kick", case=False, na=False)
    is_throw_in = play_pattern.str.contains("Throw In", case=False, na=False)

    situation[is_throw_in] = "throw_in_set_piece"
    situation[is_fk] = "free_kick_set_piece"
    situation[is_corner] = "corner_set_piece"
    situation[is_counter] = "counter_attack"
    situation[is_penalty] = "penalty"

    situation_simple = pd.Series("open_play", index=df.index)
    situation_simple[is_throw_in | is_fk | is_corner] = "set_piece"
    situation_simple[is_counter] = "counter_attack"
    situation_simple[is_penalty] = "penalty"

    return pd.DataFrame({
        "play_situation_detailed": situation,
        "play_situation": situation_simple,
        "is_open_play": (situation_simple == "open_play").astype(int),
        "is_set_piece": (situation_simple == "set_piece").astype(int),
        "is_counter": (situation_simple == "counter_attack").astype(int),
        "is_penalty": (situation_simple == "penalty").astype(int),
    }, index=df.index)


def extract_assistance_features(
    events_df: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extracts pass assistance features by joining shots with their preceding key pass.
    """
    id_col = "id" if "id" in events_df.columns else "event_id"
    kp_key = "shot_key_pass_id"

    assist_features = pd.DataFrame(index=shots_df.index)
    assist_features["assistance_type"] = "unassisted"
    assist_features["is_assisted"] = 0
    assist_features["assist_is_cross"] = 0
    assist_features["assist_is_through_ball"] = 0
    assist_features["assist_is_cutback"] = 0
    assist_features["assist_pass_length"] = np.nan
    assist_features["assist_pass_angle"] = np.nan

    if kp_key not in shots_df.columns or id_col not in events_df.columns:
        return assist_features

    valid_kp_ids = shots_df[kp_key].dropna().unique()
    if len(valid_kp_ids) == 0:
        return assist_features

    passes = events_df[(events_df["type"] == "Pass") & (events_df[id_col].isin(valid_kp_ids))].copy()

    pass_cross = passes["pass_cross"].fillna(False) if "pass_cross" in passes.columns else pd.Series(False, index=passes.index)
    pass_tb = passes["pass_through_ball"].fillna(False) if "pass_through_ball" in passes.columns else pd.Series(False, index=passes.index)
    pass_cb = passes["pass_cut_back"].fillna(False) if "pass_cut_back" in passes.columns else pd.Series(False, index=passes.index)
    pass_type = passes["pass_type"].astype(str) if "pass_type" in passes.columns else pd.Series("", index=passes.index)
    pass_len = passes["pass_length"] if "pass_length" in passes.columns else pd.Series(np.nan, index=passes.index)
    pass_ang = passes["pass_angle"] if "pass_angle" in passes.columns else pd.Series(np.nan, index=passes.index)

    def determine_assist_type(is_cr, is_t, is_c, p_type):
        if bool(is_cr):
            return "cross"
        if bool(is_t):
            return "through_ball"
        if bool(is_c):
            return "cutback"
        if p_type in ["Corner", "Free Kick", "Throw-in"]:
            return "set_piece_pass"
        if p_type == "Recovery":
            return "recovery_pass"
        return "standard_pass"

    assist_type_series = [
        determine_assist_type(c, t, cb, pt)
        for c, t, cb, pt in zip(pass_cross, pass_tb, pass_cb, pass_type)
    ]

    kp_lookup = pd.DataFrame({
        id_col: passes[id_col].values,
        "_assist_type": assist_type_series,
        "_is_cross": pass_cross.astype(bool).astype(int).values,
        "_is_tb": pass_tb.astype(bool).astype(int).values,
        "_is_cb": pass_cb.astype(bool).astype(int).values,
        "_pass_len": pass_len.values,
        "_pass_ang": pass_ang.values,
    }).drop_duplicates(subset=[id_col])

    merged = shots_df[[kp_key]].reset_index().merge(
        kp_lookup, left_on=kp_key, right_on=id_col, how="left"
    ).set_index("index")

    has_assist = merged["_assist_type"].notna()
    assist_features.loc[has_assist, "assistance_type"] = merged.loc[has_assist, "_assist_type"]
    assist_features.loc[has_assist, "is_assisted"] = 1
    assist_features.loc[has_assist, "assist_is_cross"] = merged.loc[has_assist, "_is_cross"]
    assist_features.loc[has_assist, "assist_is_through_ball"] = merged.loc[has_assist, "_is_tb"]
    assist_features.loc[has_assist, "assist_is_cutback"] = merged.loc[has_assist, "_is_cb"]
    assist_features.loc[has_assist, "assist_pass_length"] = merged.loc[has_assist, "_pass_len"]
    assist_features.loc[has_assist, "assist_pass_angle"] = merged.loc[has_assist, "_pass_ang"]

    return assist_features


def extract_contextual_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts binary flags indicating shot context.
    """
    first_time = df["shot_first_time"].fillna(False).astype(bool).astype(int) if "shot_first_time" in df.columns else pd.Series(0, index=df.index)
    one_on_one = df["shot_one_on_one"].fillna(False).astype(bool).astype(int) if "shot_one_on_one" in df.columns else pd.Series(0, index=df.index)
    open_goal = df["shot_open_goal"].fillna(False).astype(bool).astype(int) if "shot_open_goal" in df.columns else pd.Series(0, index=df.index)
    under_pressure = df["under_pressure"].fillna(False).astype(bool).astype(int) if "under_pressure" in df.columns else pd.Series(0, index=df.index)
    follows_dribble = df["shot_follows_dribble"].fillna(False).astype(bool).astype(int) if "shot_follows_dribble" in df.columns else pd.Series(0, index=df.index)

    technique = df["shot_technique"].astype(str) if "shot_technique" in df.columns else pd.Series("Normal", index=df.index)
    is_volley = technique.str.contains("Volley", case=False, na=False).astype(int)

    return pd.DataFrame({
        "is_first_time": first_time,
        "is_one_on_one": one_on_one,
        "is_open_goal": open_goal,
        "under_pressure": under_pressure,
        "follows_dribble": follows_dribble,
        "is_volley": is_volley,
    }, index=df.index)


# ==============================================================================
# 3. DEFENSIVE PRESSURE & FREEZE FRAME FEATURES
# ==============================================================================

def is_point_in_triangle(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> bool:
    """
    Cross-product test for point inside triangle.
    """
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)

    has_neg = (d1 < -1e-7) or (d2 < -1e-7) or (d3 < -1e-7)
    has_pos = (d1 > 1e-7) or (d2 > 1e-7) or (d3 > 1e-7)

    return not (has_neg and has_pos)


def extract_freeze_frame_features(
    shots_df: pd.DataFrame,
    post_left: Tuple[float, float] = POST_LEFT,
    post_right: Tuple[float, float] = POST_RIGHT,
    goal_center: Tuple[float, float] = (GOAL_CENTER_X, GOAL_CENTER_Y),
) -> pd.DataFrame:
    """
    Parses the StatsBomb 'shot_freeze_frame' array to quantify defensive structure
    and goalkeeper positioning at the moment of the shot.
    """
    shot_x, shot_y = extract_shot_coordinates(shots_df)
    has_ff = "shot_freeze_frame" in shots_df.columns

    n_shots = len(shots_df)
    cone_all: List[float] = [np.nan] * n_shots
    cone_outfield: List[float] = [np.nan] * n_shots
    min_def_dist: List[float] = [np.nan] * n_shots
    dist_to_gk: List[float] = [np.nan] * n_shots
    gk_goal_dist: List[float] = [np.nan] * n_shots
    gk_in_cone: List[float] = [np.nan] * n_shots
    gk_x_list: List[float] = [np.nan] * n_shots
    gk_y_list: List[float] = [np.nan] * n_shots
    gk_depth_list: List[float] = [np.nan] * n_shots
    gk_dev_list: List[float] = [np.nan] * n_shots

    if not has_ff:
        logger.warning("'shot_freeze_frame' column not found in shots DataFrame.")
        return pd.DataFrame({
            "defenders_in_cone": cone_all,
            "defenders_in_cone_outfield": cone_outfield,
            "distance_to_nearest_defender": min_def_dist,
            "distance_to_goalkeeper": dist_to_gk,
            "goalkeeper_x": gk_x_list,
            "goalkeeper_y": gk_y_list,
            "goalkeeper_distance_to_goal": gk_goal_dist,
            "gk_depth_from_line": gk_depth_list,
            "gk_line_deviation": gk_dev_list,
            "goalkeeper_in_cone": gk_in_cone,
        }, index=shots_df.index)

    p_lx, p_ly = post_left
    p_rx, p_ry = post_right
    g_cx, g_cy = goal_center

    for i, (ff, sx, sy) in enumerate(zip(shots_df["shot_freeze_frame"], shot_x, shot_y)):
        if ff is None or not isinstance(ff, (list, np.ndarray)) or len(ff) == 0 or np.isnan(sx) or np.isnan(sy):
            continue

        tot_in_cone = 0
        outfield_in_cone = 0
        min_def = np.inf
        gk_dist = np.nan
        gk_to_goal = np.nan
        gk_cone = 0
        gk_x_val = np.nan
        gk_y_val = np.nan
        gk_depth_val = np.nan
        gk_dev_val = np.nan

        for player_info in ff:
            if player_info.get("teammate", False):
                continue

            ploc = player_info.get("location")
            if ploc is None or len(ploc) < 2:
                continue

            px, py = float(ploc[0]), float(ploc[1])
            is_gk = player_info.get("position", {}).get("name") == "Goalkeeper"
            dist_to_shooter = np.hypot(px - sx, py - sy)

            in_cone = is_point_in_triangle(px, py, sx, sy, p_lx, p_ly, p_rx, p_ry)

            if is_gk:
                gk_dist = dist_to_shooter
                gk_to_goal = np.hypot(g_cx - px, g_cy - py)
                gk_x_val = px
                gk_y_val = py
                gk_depth_val = max(0.0, 120.0 - px)
                
                # Perpendicular line deviation from (px, py) to line S(sx, sy) -> G(120, 40)
                num = abs((g_cy - sy) * px - (g_cx - sx) * py + g_cx * sy - g_cy * sx)
                den = np.hypot(g_cy - sy, g_cx - sx)
                gk_dev_val = num / (den if den > 0 else 1.0)

                if in_cone:
                    tot_in_cone += 1
                    gk_cone = 1
            else:
                if dist_to_shooter < min_def:
                    min_def = dist_to_shooter
                if in_cone:
                    tot_in_cone += 1
                    outfield_in_cone += 1

        cone_all[i] = float(tot_in_cone)
        cone_outfield[i] = float(outfield_in_cone)
        min_def_dist[i] = float(min_def) if min_def != np.inf else np.nan
        dist_to_gk[i] = float(gk_dist)
        gk_goal_dist[i] = float(gk_to_goal)
        gk_in_cone[i] = float(gk_cone)
        gk_x_list[i] = float(gk_x_val)
        gk_y_list[i] = float(gk_y_val)
        gk_depth_list[i] = float(gk_depth_val)
        gk_dev_list[i] = float(gk_dev_val)

    return pd.DataFrame({
        "defenders_in_cone": cone_all,
        "defenders_in_cone_outfield": cone_outfield,
        "distance_to_nearest_defender": min_def_dist,
        "distance_to_goalkeeper": dist_to_gk,
        "goalkeeper_x": gk_x_list,
        "goalkeeper_y": gk_y_list,
        "goalkeeper_distance_to_goal": gk_goal_dist,
        "gk_depth_from_line": gk_depth_list,
        "gk_line_deviation": gk_dev_list,
        "goalkeeper_in_cone": gk_in_cone,
    }, index=shots_df.index)


# ==============================================================================
# 4. POSSESSION SEQUENCE METRICS
# ==============================================================================

def extract_possession_features(
    events_df: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes possession-level metrics leading up to each shot.
    """
    required_cols = {"match_id", "possession", "minute", "second"}
    if not required_cols.issubset(events_df.columns) or not required_cols.issubset(shots_df.columns):
        logger.warning("Missing required possession sequence columns.")
        return pd.DataFrame({
            "possession_duration": np.nan,
            "possession_event_count": np.nan,
            "possession_pass_count": np.nan,
            "possession_start_x": np.nan,
            "possession_progressed_distance": np.nan,
            "possession_directness": np.nan,
        }, index=shots_df.index)

    id_col = "id" if "id" in events_df.columns else "event_id"

    poss_summary = events_df.groupby(["match_id", "possession"]).agg(
        possession_start_minute=("minute", "min"),
        possession_start_second=("second", "min"),
        possession_event_count=(id_col, "count"),
        possession_pass_count=("type", lambda s: (s == "Pass").sum()),
    ).reset_index()

    if "location" in events_df.columns:
        valid_locs = events_df[events_df["location"].notna()][["match_id", "possession", "location"]].drop_duplicates(
            subset=["match_id", "possession"], keep="first"
        ).copy()
        valid_locs["possession_start_x"] = valid_locs["location"].apply(
            lambda l: float(l[0]) if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 1 else np.nan
        )
        valid_locs = valid_locs[["match_id", "possession", "possession_start_x"]]
        poss_summary = poss_summary.merge(valid_locs, on=["match_id", "possession"], how="left")
    else:
        poss_summary["possession_start_x"] = np.nan

    shot_x, _ = extract_shot_coordinates(shots_df)
    shots_temp = shots_df[["match_id", "possession", "minute", "second"]].copy()
    shots_temp["_shot_x"] = shot_x.values

    merged = shots_temp.reset_index().merge(
        poss_summary, on=["match_id", "possession"], how="left"
    ).set_index("index")

    duration = (merged["minute"] - merged["possession_start_minute"]) * 60 + (
        merged["second"] - merged["possession_start_second"]
    )
    duration = duration.clip(lower=0.0)

    progressed = merged["_shot_x"] - merged["possession_start_x"]
    directness = progressed / np.maximum(duration, 1.0)

    return pd.DataFrame({
        "possession_duration": duration.values,
        "possession_event_count": merged["possession_event_count"].values,
        "possession_pass_count": merged["possession_pass_count"].values,
        "possession_start_x": merged["possession_start_x"].values,
        "possession_progressed_distance": progressed.values,
        "possession_directness": directness.values,
    }, index=shots_df.index)


# ==============================================================================
# 5. TEMPORAL & SCORELINE FEATURES
# ==============================================================================

def calculate_scoreline_features(
    events_df: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates pre-shot game minute and match score differential.
    """
    id_col = "id" if "id" in events_df.columns else "event_id"
    is_goal_shot = (events_df["type"] == "Shot") & (events_df["shot_outcome"] == "Goal")

    timeline = events_df[[id_col, "match_id", "minute", "second", "team_id", "type"]].copy()
    timeline["is_goal"] = is_goal_shot.astype(int)

    score_records = {}
    for m_id, group in timeline.groupby("match_id", sort=False):
        teams = group["team_id"].unique()
        if len(teams) < 2:
            continue
        t1, t2 = teams[0], teams[1]

        g1 = ((group["team_id"] == t1) & (group["is_goal"] == 1)).astype(int)
        g2 = ((group["team_id"] == t2) & (group["is_goal"] == 1)).astype(int)

        s1 = g1.cumsum().shift(1, fill_value=0)
        s2 = g2.cumsum().shift(1, fill_value=0)

        for idx, t_id, sc1, sc2 in zip(group[id_col], group["team_id"], s1, s2):
            team_sc = sc1 if t_id == t1 else sc2
            opp_sc = sc2 if t_id == t1 else sc1
            score_records[idx] = (team_sc, opp_sc, team_sc - opp_sc)

    shot_id_col = "event_id" if "event_id" in shots_df.columns else id_col
    team_scores = [score_records.get(i, (0, 0, 0))[0] for i in shots_df[shot_id_col]]
    opp_scores = [score_records.get(i, (0, 0, 0))[1] for i in shots_df[shot_id_col]]
    diffs = [score_records.get(i, (0, 0, 0))[2] for i in shots_df[shot_id_col]]

    score_state = pd.Series("drawing", index=shots_df.index)
    score_state[np.array(diffs) > 0] = "leading"
    score_state[np.array(diffs) < 0] = "trailing"

    return pd.DataFrame({
        "game_minute": shots_df["minute"].values if "minute" in shots_df.columns else np.nan,
        "team_score": team_scores,
        "opponent_score": opp_scores,
        "score_differential": diffs,
        "score_state": score_state.values,
        "is_leading": (score_state == "leading").astype(int).values,
        "is_drawing": (score_state == "drawing").astype(int).values,
        "is_trailing": (score_state == "trailing").astype(int).values,
    }, index=shots_df.index)


# ==============================================================================
# 6. MASTER FEATURE EXTRACTION PIPELINE
# ==============================================================================

def build_xg_features(
    events_df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Master pipeline: extracts all shots from raw StatsBomb events and constructs
    the complete, analysis-ready Expected Goals (xG) feature dataset.
    """
    logger.info("Extracting shots from event dataset...")
    shots = events_df[events_df["type"] == "Shot"].copy()

    if len(shots) == 0:
        logger.warning("No shot events found in the provided DataFrame.")
        return pd.DataFrame()

    logger.info(f"Found {len(shots)} shots. Engineering features...")

    # 1. Coordinates and Geometric Features (including Day 18 features)
    shot_x, shot_y = extract_shot_coordinates(shots)
    geom_features = calculate_geometric_features(shot_x.values, shot_y.values)
    geom_features.index = shots.index

    # 2. Contextual & Technical Features
    bp_features = extract_body_part_features(shots)
    sit_features = extract_play_situation_features(shots)
    assist_features = extract_assistance_features(events_df, shots)
    flag_features = extract_contextual_flags(shots)

    # 3. Defensive Pressure & Freeze Frame (including GK coordinates and positioning)
    ff_features = extract_freeze_frame_features(shots)

    # 4. Visible Goal Angle (occlusion-adjusted)
    visible_features = calculate_visible_goal_angle(
        geom_features["shot_angle"].values,
        ff_features["distance_to_goalkeeper"].values,
        ff_features["goalkeeper_in_cone"].values,
    )
    visible_features.index = shots.index

    # 5. Possession Sequence
    poss_features = extract_possession_features(events_df, shots)

    # 6. Temporal & Scoreline Features
    id_col = "id" if "id" in shots.columns else "event_id"
    meta_cols = {
        "event_id": shots[id_col].values if id_col in shots.columns else np.arange(len(shots)),
        "match_id": shots["match_id"].values if "match_id" in shots.columns else np.nan,
        "minute": shots["minute"].values if "minute" in shots.columns else np.nan,
        "second": shots["second"].values if "second" in shots.columns else np.nan,
        "player_id": shots["player_id"].values if "player_id" in shots.columns else np.nan,
        "player_name": shots["player"].apply(lambda p: p.get("name") if isinstance(p, dict) else str(p)).values if "player" in shots.columns else np.nan,
        "team_id": shots["team_id"].values if "team_id" in shots.columns else np.nan,
        "team_name": shots["team"].apply(lambda t: t.get("name") if isinstance(t, dict) else str(t)).values if "team" in shots.columns else np.nan,
    }
    meta_df = pd.DataFrame(meta_cols, index=shots.index)

    temporal_features = calculate_scoreline_features(events_df, meta_df)

    # Target & Benchmark xG
    target_df = pd.DataFrame(index=shots.index)
    if include_target:
        outcome = shots["shot_outcome"].astype(str) if "shot_outcome" in shots.columns else pd.Series("", index=shots.index)
        target_df["shot_outcome"] = outcome
        target_df["is_goal"] = (outcome.str.lower() == "goal").astype(int)

        if "shot_statsbomb_xg" in shots.columns:
            target_df["statsbomb_xg"] = shots["shot_statsbomb_xg"].astype(float)
        elif "statsbomb_xg" in shots.columns:
            target_df["statsbomb_xg"] = shots["statsbomb_xg"].astype(float)

    # Concatenate all feature blocks
    feature_df = pd.concat([
        meta_df,
        geom_features,
        visible_features,
        bp_features,
        sit_features,
        assist_features,
        flag_features,
        ff_features,
        poss_features,
        temporal_features,
        target_df,
    ], axis=1)

    logger.info(f"Successfully engineered {feature_df.shape[1]} features across {feature_df.shape[0]} shots.")
    return feature_df


def main():
    """
    Self-test execution script for xG feature engineering.
    """
    import os

    raw_path = "data/raw/statsbomb/sb_events_43_106.parquet"
    out_dir = "data/processed/football"
    out_path = os.path.join(out_dir, "xg_features_sb_43_106.parquet")

    if not os.path.exists(raw_path):
        logger.error(f"Sample data file not found at: {raw_path}")
        return

    logger.info(f"Loading raw StatsBomb events from {raw_path}...")
    events = pd.read_parquet(raw_path)

    features = build_xg_features(events)
    logger.info(f"Extracted shape: {features.shape}")

    os.makedirs(out_dir, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info(f"Saved engineered features to {out_path}")

    print("\n--- Top Features Overview ---")
    cols_to_show = [
        "distance_to_goal", "distance_to_nearest_post", "visible_angle_to_goal_deg",
        "is_central_shot", "gk_line_deviation", "score_state", "is_goal"
    ]
    print(features[cols_to_show].head(8))
    print("\n--- Summary Statistics ---")
    print(features[["distance_to_goal", "visible_angle_to_goal_deg", "gk_line_deviation"]].describe().round(2))


if __name__ == "__main__":
    main()
