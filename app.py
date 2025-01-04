import streamlit as st
import pandas as pd
import gpxpy
import folium
from folium import plugins
from streamlit_folium import st_folium
import os
import base64
import bisect
from datetime import datetime, timedelta, date
import calendar
import random

# Set page layout to wide
st.set_page_config(layout='wide')

########################################
# 1. TOP SECTION (PHOTO & TITLE)
########################################
user_image_path = "images/GermanAlonzo.jpg"
if os.path.exists(user_image_path):
    with open(user_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    user_photo_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/jpeg;base64,{base64_image}"
            alt="User Photo"
            style="
                border-radius: 50%;
                width: 120px;
                height: 120px;
                margin-right: 15px;
                object-fit: cover;
                transform: scale(1.2);">
        <h1 style="margin: 0;">German Alonzo Running App</h1>
    </div>
    """
    st.markdown(user_photo_html, unsafe_allow_html=True)
else:
    st.title("German Alonzo Running App")

########################################
# 2. HELPER FUNCTIONS
########################################

def parse_gpx(file):
    """
    Reads a GPX file and returns a DataFrame of lat, lon, elevation, and time.
    """
    with open(file, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time,
                })
    return pd.DataFrame(data)

def filter_gpx_data(df):
    """
    Removes data points with large time gaps, unrealistic speeds,
    and discards entire file if total time/distance ratio is too large.
    """
    if df.empty:
        return df

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['time_diff'] = df['time'].diff().dt.total_seconds()

    # Distance between consecutive points (approx. in km)
    df['distance'] = (
        df[['latitude', 'longitude']]
        .diff()
        .pow(2)
        .sum(axis=1)
        .pow(0.5) * 111
    )

    # Speed in km/h
    df['speed_kmh'] = df['distance'] / (df['time_diff'] / 3600)

    # Filtering thresholds
    PAUSE_THRESHOLD_SECONDS = 120  # 2 minutes
    MAX_RUNNING_SPEED_KMH = 20     # 20 km/h

    df = df[
        (df['time_diff'] <= PAUSE_THRESHOLD_SECONDS) &
        (df['speed_kmh'] <= MAX_RUNNING_SPEED_KMH) &
        (df['speed_kmh'] > 0)
    ]
    df.dropna(inplace=True)

    if df.empty:
        return df

    # If avg speed < 6 km/h => discard
    total_distance = df['distance'].sum()
    total_time = (df['time'].max() - df['time'].min()).total_seconds()
    MIN_AVG_SPEED_KM_H = 6
    max_time_threshold = total_distance / MIN_AVG_SPEED_KM_H * 3600
    if total_time > max_time_threshold:
        return pd.DataFrame()

    return df

def calculate_best_pace_and_time(df, target_distance):
    """
    Returns (best_pace_str, best_time_str) for covering at least 'target_distance' km.
    'best_pace_str' = mm:ss / km
    'best_time_str' = HH:MM:SS total time for that distance
    If not found => ("N/A", "N/A").
    """
    if df.empty:
        return ("N/A", "N/A")

    df_sorted = df.sort_values(by='time').reset_index(drop=True)
    cumDist = df_sorted['distance'].cumsum().tolist()
    times = df_sorted['time'].tolist()
    n = len(df_sorted)

    best_time_seconds = None

    for i in range(n):
        dist_at_i = cumDist[i - 1] if i > 0 else 0.0
        start_time = times[i]

        needed_distance = dist_at_i + target_distance
        j = bisect.bisect_left(cumDist, needed_distance, i, n)
        if j < n:
            segment_time_seconds = (times[j] - start_time).total_seconds()
            if best_time_seconds is None or segment_time_seconds < best_time_seconds:
                best_time_seconds = segment_time_seconds

    if best_time_seconds is None:
        return ("N/A", "N/A")

    # Pace & total time
    pace_per_km = best_time_seconds / target_distance
    pace_minutes, pace_seconds = divmod(int(pace_per_km), 60)
    pace_str = f"{pace_minutes}m {pace_seconds}s"

    hh = best_time_seconds // 3600
    mm = (best_time_seconds % 3600) // 60
    ss = int(best_time_seconds % 60)
    best_time_str = f"{int(hh)}h {int(mm)}m {int(ss)}s"

    return (pace_str, best_time_str)

def parse_pace_string(pace_str):
    """
    Pace string like '5m 15s' => returns float(5.25) ~ 5.25 min/km
    If pace_str='N/A', return a big number (like 999)
    """
    if pace_str == "N/A":
        return 999.0
    # "5m 30s" => 5 and 30
    split_data = pace_str.split("m ")
    if len(split_data) != 2:
        return 999.0
    min_part = split_data[0]
    sec_part = split_data[1].replace("s", "")
    try:
        minutes = float(min_part)
        seconds = float(sec_part)
        total = minutes + seconds/60.0
        return total
    except:
        return 999.0

########################################
# NEW: variable intervals each training day
########################################
def get_variable_interval(day_of_month, five_k_pace):
    """
    Return a different interval suggestion each training day
    and adapt to best 5K pace (advanced / intermediate / beginner).
    """
    # Pools of intervals for each level
    advanced_intervals = [
        "6 x 800m at fast pace (2' rest)",
        "8 x 400m speed intervals (1' rest)",
        "5 x 1000m tempo intervals (2'30 rest)",
    ]
    intermediate_intervals = [
        "4 x 800m moderate pace (2' rest)",
        "6 x 400m moderate intervals (1'30 rest)",
        "3 x 1000m moderate intervals (3' rest)",
    ]
    beginner_intervals = [
        "4 x 600m easy pace (1'30 rest)",
        "3 x 800m comfortable pace (2' rest)",
        "6 x 200m short intervals (1' rest)",
    ]

    # Decide level
    if five_k_pace < 5.0:
        pool = advanced_intervals
    elif five_k_pace < 6.0:
        pool = intermediate_intervals
    else:
        pool = beginner_intervals

    # Cycle through the pool based on day_of_month
    idx = (day_of_month - 1) % len(pool)
    return pool[idx]

########################################
# 3. MAIN APP LOGIC
########################################

gpx_folder = "GPX"
now = datetime.now()
thirty_days_ago = now - timedelta(days=30)

combined_df = pd.DataFrame()
if os.path.exists(gpx_folder):
    for file_name in os.listdir(gpx_folder):
        if file_name.endswith(".gpx"):
            file_path = os.path.join(gpx_folder, file_name)
            raw_df = parse_gpx(file_path)

            # Convert to UTC => strip tz => naive
            raw_df['time'] = pd.to_datetime(raw_df['time'], utc=True, errors='coerce').dt.tz_localize(None)
            # Keep only last 30 days
            raw_df = raw_df[raw_df['time'] >= thirty_days_ago]

            if not raw_df.empty:
                filtered_df = filter_gpx_data(raw_df)
                if not filtered_df.empty:
                    combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)

    if combined_df.empty:
        st.write("No valid GPX data from the last 30 days after filtering.")
    else:
        # Sort
        combined_df['time'] = pd.to_datetime(combined_df['time'], errors='coerce')
        combined_df.sort_values(by='time', inplace=True, ignore_index=True)

        # 1) DATE SELECTION
        st.header("Select a Date (Last 30 Days)")
        min_date = combined_df['time'].dt.date.min()
        max_date = combined_df['time'].dt.date.max()
        last_run_date = combined_df['time'].dt.date.max()
        selected_date = st.date_input(
            "Choose a date",
            value=last_run_date,
            min_value=min_date,
            max_value=max_date
        )

        filtered_df = combined_df[combined_df['time'].dt.date == selected_date]

        # 2) ROUTE MAP
        st.subheader("Route Map")
        if not filtered_df.empty:
            start_coords = (filtered_df['latitude'].iloc[0], filtered_df['longitude'].iloc[0])
            m = folium.Map(location=start_coords, zoom_start=14, tiles="CartoDB Dark_Matter")

            plugins.AntPath(
                locations=filtered_df[['latitude', 'longitude']].values,
                dash_array=[20, 20],
                delay=1000,
                color='red',
                pulse_color='orange',
                weight=5,
            ).add_to(m)

            col1, col2 = st.columns([2, 1])
            with col1:
                st_folium(m, width=750, height=450)
            with col2:
                total_distance = filtered_df['distance'].sum()
                min_time = filtered_df['time'].min()
                max_time = filtered_df['time'].max()
                overall_time = max_time - min_time

                total_seconds = int(overall_time.total_seconds())
                hh = total_seconds // 3600
                mm = (total_seconds % 3600) // 60
                ss = total_seconds % 60

                if total_distance > 0:
                    pace_sec_per_km = total_seconds / total_distance
                    pace_mm, pace_ss = divmod(int(pace_sec_per_km), 60)
                else:
                    pace_mm, pace_ss = (0, 0)

                st.subheader("📊 Details")
                st.markdown(f"""
                - **🏃‍♂️ Distance:** {total_distance:.2f} km  
                - **⏱️ Pace:** {pace_mm}m {pace_ss}s / km  
                - **⌛ Overall Time:** {hh}h {mm}m {ss}s  
                """, unsafe_allow_html=True)
        else:
            st.write("No data available for the selected date.")

        # 3) BEST 5K / 10K
        st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)
        st.subheader("🏅 Best Paces and Timing")

        best_5k_pace_str, best_5k_time_str = calculate_best_pace_and_time(combined_df, 5)
        best_10k_pace_str, best_10k_time_str = calculate_best_pace_and_time(combined_df, 10)

        data_for_table = [
            {"Distance": "5 km", "Best Pace": best_5k_pace_str, "Best Time": best_5k_time_str},
            {"Distance": "10 km", "Best Pace": best_10k_pace_str, "Best Time": best_10k_time_str},
        ]
        df_best = pd.DataFrame(data_for_table)
        st.table(df_best)

        # We'll parse the best 5K pace as minutes / km
        five_k_pace_minutes = parse_pace_string(best_5k_pace_str)

        # 4) TRAINING CALENDAR
        st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)
        st.subheader("📅 This Month's Training Calendar")

        today = date.today()
        current_year = today.year
        current_month = today.month

        cal = calendar.TextCalendar(firstweekday=0)
        month_days = cal.monthdayscalendar(current_year, current_month)
        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # CSS: White letters, dark blue background; sparkle Saturdays; variable intervals Tue/Thu
        table_html = """
        <style>
        .calendar-table {
            border-collapse: collapse;
            margin: 0 auto;
        }
        .calendar-table th, .calendar-table td {
            border: 1px solid #ddd;
            text-align: center;
            width: 90px;
            height: 80px;
            vertical-align: middle;
        }
        /* Dark blue background, white text for the day-of-week header */
        .calendar-table th {
            background-color: #003366;
            color: #ffffff;
            font-weight: bold;
        }
        .train-day {
            background-color: #cce5ff; /* Light blue highlight for training days */
            font-weight: bold;
            color: #000;
        }
        .sparkle-day {
            background-color: #ffeeba; /* A soft highlight for Saturdays */
            font-weight: bold;
            color: #000;
        }
        .day-info {
            font-size: 0.85em;
            margin-top: 4px;
            display: block;
        }
        .empty-day {
            background-color: #fafafa;
            color: #ccc;
        }
        </style>
        <table class="calendar-table">
          <thead>
            <tr>
        """

        # Header row (Mon, Tue, Wed, Thu, Fri, Sat, Sun)
        for dow in days_of_week:
            table_html += f"<th>{dow}</th>"
        table_html += "</tr></thead><tbody>"

        # Build the rows
        for week in month_days:
            table_html += "<tr>"
            for i, day_num in enumerate(week):
                if day_num == 0:
                    # outside current month
                    table_html += "<td class='empty-day'></td>"
                else:
                    # i=5 => Saturday => sparkle
                    if i == 5:
                        table_html += f"<td class='sparkle-day'>✨ {day_num}</td>"
                    # i=1 => Tuesday, i=3 => Thursday => training day
                    elif i in [1, 3]:
                        interval_suggestion = get_variable_interval(day_num, five_k_pace_minutes)
                        table_html += (
                            f"<td class='train-day'>"
                            f"{day_num}"
                            f"<br><span class='day-info'>{interval_suggestion}</span>"
                            f"</td>"
                        )
                    else:
                        table_html += f"<td>{day_num}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        st.markdown(table_html, unsafe_allow_html=True)

else:
    st.write(f"Folder '{gpx_folder}' does not exist. Please create it and add your GPX files.")
