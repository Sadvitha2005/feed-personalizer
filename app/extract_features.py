from datetime import datetime

def parse_hour_minute(timestamp):
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        return dt.hour, dt.minute, dt
    except ValueError:
        return -1, -1, None

def time_to_minutes(hour, minute):
    return hour * 60 + minute

def compute_time_match_score(post_hour, post_minute, active_ranges):
    post_minutes = time_to_minutes(post_hour, post_minute)
    min_distance = float("inf")

    for time_range in active_ranges:
        try:
            start_str, end_str = time_range.split("-")
            start_hour, start_minute = map(int, start_str.split(":"))
            end_hour, end_minute = map(int, end_str.split(":"))
            start_minutes = time_to_minutes(start_hour, start_minute)
            end_minutes = time_to_minutes(end_hour, end_minute)
        except ValueError:
            continue

        if start_minutes <= end_minutes:
            if start_minutes <= post_minutes <= end_minutes:
                return 1.0
        else:
            if post_minutes >= start_minutes or post_minutes <= end_minutes:
                return 1.0

        distance_to_start = min(abs(post_minutes - start_minutes), 1440 - abs(post_minutes - start_minutes))
        distance_to_end = min(abs(post_minutes - end_minutes), 1440 - abs(post_minutes - end_minutes))
        min_distance = min(min_distance, distance_to_start, distance_to_end)

    return round(max(0.0, 1 - (min_distance / 600)), 2)

def get_post_hour_bucket(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def get_time_periods(dt):
    if dt.weekday() < 5:
        return "Weekday"
    else:
        return "Weekend"

def get_karma_bucket(karma):
    if karma <= 33:
        return "low"
    elif karma <= 66:
        return "medium"
    else:
        return "high"

def safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0

def extract_features(post, user_profile):
    user_followed_tags = user_profile.get("tags_followed", [])
    buddies = user_profile.get("buddies", [])
    active_hours = user_profile.get("active_hours", [])

    post_tags = post.get("tags", [])
    author_id = post.get("author_id")
    content_type = post.get("content_type", "unknown")
    karma = post.get("karma", 0)

    # Time features
    post_hour, post_minute, dt = parse_hour_minute(post.get("created_at", ""))
    if post_hour != -1:
        time_match_score = compute_time_match_score(post_hour, post_minute, active_hours)
        post_hour_bucket = get_post_hour_bucket(post_hour)
        time_periods = get_time_periods(dt)
    else:
        time_match_score = 0.0
        post_hour_bucket = "Unknown"
        time_periods = "Unknown"

    # Basic booleans
    user_follows_tag = any(tag in user_followed_tags for tag in post_tags)
    is_buddy_post = author_id in buddies
    karma_bucket = get_karma_bucket(karma)

    return {
        "karma": karma,
        "time_match_score": time_match_score,
        "user_follows_tag": user_follows_tag,
        "is_buddy_post": is_buddy_post,
        "Post Type": content_type,
        "Weekday Type": time_periods,
        "Time Periods": post_hour_bucket,
        "karma_bucket": karma_bucket
    }
