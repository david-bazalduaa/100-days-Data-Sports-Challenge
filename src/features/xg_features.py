"""
Day 17: Feature Engineering for Expected Goals (xG) Model (Part 1)

This module provides modular, reusable, and production-grade feature engineering
functions for football (soccer) event data based on the StatsBomb coordinate system (120x80).

Feature Categories:
1. Geometric Features:
   - Euclidean distance to goal center (120, 40)
   - Angle between goal posts (120, 36) and (120, 44)
   - Distances to left and right posts
2. Contextual & Technical Features:
   - Body part (right foot, left foot, head, other)
   - Play situation (open play, set piece, counter attack, penalty)
   - Assistance type from preceding key pass (cross, through ball, cutback, recovery, set piece, unassisted)
   - Context flags (first time shot, follows dribble, open goal, one on one, under pressure)
3. Defensive Pressure & Freeze Frame Features:
   - Opponents inside the triangular shot cone (defenders_in_cone)
   - Distance to nearest outfield defender
   - Distance to goalkeeper and goalkeeper distance to goal
4. Possession Sequence Features:
   - Possession sequence duration (seconds)
   - Number of passes in possession leading to shot
   - Total events in sequence and pitch progression (start_x to shot_x)
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
        # If populated, return directly
        if shot_x.notna().sum() > 0:
            return shot_x, shot_y

    if "location" in df.columns:
        # Extract from list/array [x, y]
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

    Formula:
        distance = sqrt((goal_x - shot_x)^2 + (goal_y - shot_y)^2)

    Args:
        shot_x: X-coordinate(s) of the shot location.
        shot_y: Y-coordinate(s) of the shot location.
        goal_x: Goal center X-coordinate (default: 120.0).
        goal_y: Goal center Y-coordinate (default: 40.0).

    Returns:
        Numpy array of distances.
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

    Formula:
        theta = |atan2(post_right_y - y, post_right_x - x) - atan2(post_left_y - y, post_left_x - x)|

    Args:
        shot_x: X-coordinate(s) of the shot.
        shot_y: Y-coordinate(s) of the shot.
        post_left: (x, y) coordinates of the left post (default: (120.0, 36.0)).
        post_right: (x, y) coordinates of the right post (default: (120.0, 44.0)).
        in_degrees: If True, returns angle in degrees; otherwise in radians.

    Returns:
        Numpy array of shot angles.
    """
    sx = np.asarray(shot_x, dtype=float)
    sy = np.asarray(shot_y, dtype=float)

    angle_right = np.arctan2(post_right[1] - sy, post_right[0] - sx)
    angle_left = np.arctan2(post_left[1] - sy, post_left[0] - sx)

    shot_angle = np.abs(angle_right - angle_left)

    # Normalize to [0, pi] if needed
    shot_angle = np.where(shot_angle > np.pi, 2 * np.pi - shot_angle, shot_angle)

    if in_degrees:
        return np.degrees(shot_angle)
    return shot_angle


def calculate_geometric_features(
    shot_x: np.ndarray,
    shot_y: np.ndarray,
) -> pd.DataFrame:
    """
    Computes a comprehensive set of geometric features for shots.

    Args:
        shot_x: Array of shot X-coordinates.
        shot_y: Array of shot Y-coordinates.

    Returns:
        DataFrame containing geometric features.
    """
    sx = np.asarray(shot_x, dtype=float)
    sy = np.asarray(shot_y, dtype=float)

    dist_to_center = calculate_shot_distance(sx, sy, GOAL_CENTER_X, GOAL_CENTER_Y)
    dist_to_left_post = calculate_shot_distance(sx, sy, POST_LEFT[0], POST_LEFT[1])
    dist_to_right_post = calculate_shot_distance(sx, sy, POST_RIGHT[0], POST_RIGHT[1])

    angle_rad = calculate_shot_angle(sx, sy, POST_LEFT, POST_RIGHT, in_degrees=False)
    angle_deg = np.degrees(angle_rad)

    # Shot eccentricity: deviation from the pitch center axis (y = 40)
    eccentricity = np.abs(sy - GOAL_CENTER_Y)

    return pd.DataFrame({
        "shot_x": sx,
        "shot_y": sy,
        "distance_to_goal": dist_to_center,
        "distance_to_left_post": dist_to_left_post,
        "distance_to_right_post": dist_to_right_post,
        "shot_angle": angle_rad,
        "shot_angle_deg": angle_deg,
        "shot_eccentricity": eccentricity,
    })


# ==============================================================================
# 2. CONTEXTUAL & TECHNICAL FEATURES
# ==============================================================================

def extract_body_part_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes and encodes shot body part features.

    Categories:
    - 'right_foot'
    - 'left_foot'
    - 'head'
    - 'other'

    Args:
        df: Shots DataFrame containing 'shot_body_part'.

    Returns:
        DataFrame with standardized body part category and indicator flags.
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
    Determines the play situation (game state context) for each shot.

    Hierarchy:
    1. Penalty: `shot_type == 'Penalty'`
    2. Counter Attack: `play_pattern == 'From Counter'`
    3. Corner: `play_pattern == 'From Corner'` or `shot_type == 'Corner'`
    4. Free Kick: `play_pattern == 'From Free Kick'` or `shot_type == 'Free Kick'`
    5. Throw-in: `play_pattern == 'From Throw In'`
    6. Open Play: `play_pattern == 'Regular Play'` or `shot_type == 'Open Play'`

    Args:
        df: Shots DataFrame containing 'shot_type' and/or 'play_pattern'.

    Returns:
        DataFrame with detailed and simplified play situation columns.
    """
    shot_type = df["shot_type"].astype(str) if "shot_type" in df.columns else pd.Series([""] * len(df), index=df.index)
    play_pattern = df["play_pattern"].astype(str) if "play_pattern" in df.columns else pd.Series([""] * len(df), index=df.index)

    situation = pd.Series("open_play", index=df.index)

    # Specific conditions
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

    # Simplified 4-category classification for modeling
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
    StatsBomb matches key passes via `shot_key_pass_id == pass.id`.

    Categories of `assistance_type`:
    - 'unassisted' (no preceding key pass)
    - 'cross' (assisting pass was a cross)
    - 'through_ball' (assisting pass was a through ball)
    - 'cutback' (assisting pass was a cutback)
    - 'set_piece_pass' (assisting pass from corner, free kick, throw-in)
    - 'recovery_pass' (assisting pass following an immediate recovery)
    - 'standard_pass' (open play regular assist)

    Args:
        events_df: Full StatsBomb events DataFrame containing passes.
        shots_df: Shots DataFrame containing 'shot_key_pass_id'.

    Returns:
        DataFrame of assistance-related features matching shots_df's index.
    """
    id_col = "id" if "id" in events_df.columns else "event_id"
    kp_key = "shot_key_pass_id"

    # Initialize return DataFrame with default values
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

    # Filter passes that served as shot assists
    valid_kp_ids = shots_df[kp_key].dropna().unique()
    if len(valid_kp_ids) == 0:
        return assist_features

    passes = events_df[(events_df["type"] == "Pass") & (events_df[id_col].isin(valid_kp_ids))].copy()

    # Selected pass columns with safe fallback
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

    # Merge onto shots
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
    Extracts binary flags indicating shot context (technique, pressure, open goal, 1v1).

    Args:
        df: Shots DataFrame.

    Returns:
        DataFrame with indicator flags.
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
    Determines whether point P(px, py) lies strictly or on boundary inside
    triangle formed by vertices A(ax, ay), B(bx, by), C(cx, cy).
    Uses the cross-product sign test.

    Args:
        px, py: Coordinates of point to evaluate.
        ax, ay: Triangle vertex A (shooter position).
        bx, by: Triangle vertex B (left post).
        cx, cy: Triangle vertex C (right post).

    Returns:
        Boolean indicating whether P is inside triangle ABC.
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
    and goalkeeper positioning at the exact moment of the shot.

    Engineered Features:
    - defenders_in_cone: Total opponents (including GK) inside the shot cone triangle.
    - defenders_in_cone_outfield: Total outfield opponents blocking the shot cone.
    - distance_to_nearest_defender: Euclidean distance to the nearest outfield defender.
    - distance_to_goalkeeper: Euclidean distance between shooter and goalkeeper.
    - goalkeeper_distance_to_goal: Euclidean distance from goalkeeper to goal center.
    - goalkeeper_in_cone: 1 if goalkeeper is within the shot cone, else 0.

    Args:
        shots_df: DataFrame containing shots with 'location' (or 'shot_x'/'shot_y')
                  and 'shot_freeze_frame'.
        post_left: Coordinates of left post (120, 36).
        post_right: Coordinates of right post (120, 44).
        goal_center: Goal center coordinates (120, 40).

    Returns:
        DataFrame with defensive pressure and freeze frame metrics.
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

    if not has_ff:
        logger.warning("'shot_freeze_frame' column not found in shots DataFrame. Returning NaNs for freeze frame features.")
        return pd.DataFrame({
            "defenders_in_cone": cone_all,
            "defenders_in_cone_outfield": cone_outfield,
            "distance_to_nearest_defender": min_def_dist,
            "distance_to_goalkeeper": dist_to_gk,
            "goalkeeper_distance_to_goal": gk_goal_dist,
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

        for player_info in ff:
            # Only consider opponents (teammate == False)
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

    return pd.DataFrame({
        "defenders_in_cone": cone_all,
        "defenders_in_cone_outfield": cone_outfield,
        "distance_to_nearest_defender": min_def_dist,
        "distance_to_goalkeeper": dist_to_gk,
        "goalkeeper_distance_to_goal": gk_goal_dist,
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

    Features:
    - possession_duration: Total seconds from possession start to shot moment.
    - possession_event_count: Total event actions in this possession.
    - possession_pass_count: Number of completed or attempted passes in possession.
    - possession_start_x: Pitch X-coordinate where the possession originated.
    - possession_progressed_distance: Forward pitch progression (shot_x - possession_start_x).
    - possession_directness: Progression ratio (progression / duration).

    Args:
        events_df: Full StatsBomb events DataFrame for context.
        shots_df: Filtered shots DataFrame containing 'match_id', 'possession',
                  'minute', 'second', and shot coordinates.

    Returns:
        DataFrame containing possession sequence features.
    """
    required_cols = {"match_id", "possession", "minute", "second"}
    if not required_cols.issubset(events_df.columns) or not required_cols.issubset(shots_df.columns):
        logger.warning("Missing required possession sequence columns. Returning default NaNs.")
        return pd.DataFrame({
            "possession_duration": np.nan,
            "possession_event_count": np.nan,
            "possession_pass_count": np.nan,
            "possession_start_x": np.nan,
            "possession_progressed_distance": np.nan,
            "possession_directness": np.nan,
        }, index=shots_df.index)

    id_col = "id" if "id" in events_df.columns else "event_id"

    # Aggregated possession summary per match and possession index
    poss_summary = events_df.groupby(["match_id", "possession"]).agg(
        possession_start_minute=("minute", "min"),
        possession_start_second=("second", "min"),
        possession_event_count=(id_col, "count"),
        possession_pass_count=("type", lambda s: (s == "Pass").sum()),
    ).reset_index()

    # Extract origin location for each possession
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

    # Merge summary into shots_df
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
# 5. MASTER FEATURE EXTRACTION PIPELINE
# ==============================================================================

def build_xg_features(
    events_df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Master pipeline: extracts all shots from raw StatsBomb events and constructs
    the complete, analysis-ready Expected Goals (xG) feature dataset.

    Args:
        events_df: Full StatsBomb event DataFrame (e.g. from sb.events or parquet).
        include_target: If True, includes target column 'is_goal' and benchmark 'statsbomb_xg'.

    Returns:
        DataFrame containing one row per shot with all engineered features.
    """
    logger.info("Extracting shots from event dataset...")
    shots = events_df[events_df["type"] == "Shot"].copy()

    if len(shots) == 0:
        logger.warning("No shot events found in the provided DataFrame.")
        return pd.DataFrame()

    logger.info(f"Found {len(shots)} shots. Engineering features...")

    # 1. Coordinates and Geometric Features
    shot_x, shot_y = extract_shot_coordinates(shots)
    geom_features = calculate_geometric_features(shot_x.values, shot_y.values)
    geom_features.index = shots.index

    # 2. Contextual & Technical Features
    bp_features = extract_body_part_features(shots)
    sit_features = extract_play_situation_features(shots)
    assist_features = extract_assistance_features(events_df, shots)
    flag_features = extract_contextual_flags(shots)

    # 3. Defensive Pressure & Freeze Frame
    ff_features = extract_freeze_frame_features(shots)

    # 4. Possession Sequence
    poss_features = extract_possession_features(events_df, shots)

    # Metadata columns
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

    # Concatenate all components
    feature_df = pd.concat([
        meta_df,
        geom_features,
        bp_features,
        sit_features,
        assist_features,
        flag_features,
        ff_features,
        poss_features,
        target_df,
    ], axis=1)

    logger.info(f"Successfully engineered {feature_df.shape[1]} features across {feature_df.shape[0]} shots.")
    return feature_df


def main():
    """
    Self-test execution script for Day 17 feature extraction.
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

    # Display overview
    print("\n--- Feature Engineering Overview ---")
    print(features[["distance_to_goal", "shot_angle_deg", "body_part", "play_situation", "assistance_type", "defenders_in_cone", "is_goal"]].head(10))
    print("\n--- Summary Statistics ---")
    print(features[["distance_to_goal", "shot_angle_deg", "defenders_in_cone", "possession_duration"]].describe())


if __name__ == "__main__":
    main()
