import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone
from difflib import get_close_matches
from time import sleep

from requests.exceptions import ReadTimeout, RequestException

from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import (
    boxscoretraditionalv2,
    commonplayerinfo,
    leaguestandingsv3,
    leaguegamefinder,
    playergamelog,
    teamgamelog,
)
from nba_api.stats.static import players, teams


STATS_TIMEOUT = 60
RETRY_COUNT = 3
RETRY_DELAY = 2
SCORE_REQUEST_DELAY = 0.6
PLAYIN_REQUEST_DELAY = 0.6
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


def retry_request(func, *args, **kwargs):
    last_error = None

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except (ReadTimeout, RequestException) as error:
            last_error = error
            if attempt < RETRY_COUNT:
                sleep(RETRY_DELAY)

    raise last_error


def get_current_nba_season():
    today = datetime.today()
    year = today.year

    if today.month >= 8:
        start_year = year
    else:
        start_year = year - 1

    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct_from_made_attempted(made, attempted):
    if attempted == 0:
        return 0.0
    return (made / attempted) * 100.0


def format_pct(value):
    return f"{value:.1f}%"


def parse_minutes_to_decimal(value) -> float:
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes + (seconds / 60.0)
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return (hours * 60) + minutes + (seconds / 60.0)
        except ValueError:
            return 0.0

    return safe_float(text, 0.0)


def format_minutes_decimal(value: float) -> str:
    return f"{value:.1f}"


def format_minutes_display(value) -> str:
    minutes_decimal = parse_minutes_to_decimal(value)
    return f"{minutes_decimal:.1f}"


def build_average_block(games_df: pd.DataFrame) -> dict:
    avg_min = safe_float(games_df["MIN_DECIMAL"].mean())
    avg_pts = safe_float(games_df["PTS"].mean())
    avg_reb = safe_float(games_df["REB"].mean())
    avg_ast = safe_float(games_df["AST"].mean())
    avg_stl = safe_float(games_df["STL"].mean())
    avg_blk = safe_float(games_df["BLK"].mean())
    avg_tov = safe_float(games_df["TOV"].mean())

    avg_fgm = safe_float(games_df["FGM"].mean())
    avg_fga = safe_float(games_df["FGA"].mean())
    avg_ftm = safe_float(games_df["FTM"].mean())
    avg_fta = safe_float(games_df["FTA"].mean())
    avg_fg3m = safe_float(games_df["FG3M"].mean())
    avg_fg3a = safe_float(games_df["FG3A"].mean())

    total_fgm = safe_float(games_df["FGM"].sum())
    total_fga = safe_float(games_df["FGA"].sum())
    total_ftm = safe_float(games_df["FTM"].sum())
    total_fta = safe_float(games_df["FTA"].sum())
    total_fg3m = safe_float(games_df["FG3M"].sum())
    total_fg3a = safe_float(games_df["FG3A"].sum())

    avg_fg_pct = pct_from_made_attempted(total_fgm, total_fga)
    avg_ft_pct = pct_from_made_attempted(total_ftm, total_fta)
    avg_fg3_pct = pct_from_made_attempted(total_fg3m, total_fg3a)

    return {
        "min": format_minutes_decimal(avg_min),
        "pts": round(avg_pts, 1),
        "reb": round(avg_reb, 1),
        "ast": round(avg_ast, 1),
        "stl": round(avg_stl, 1),
        "blk": round(avg_blk, 1),
        "tov": round(avg_tov, 1),
        "fg": f"{avg_fgm:.1f}/{avg_fga:.1f} ({format_pct(avg_fg_pct)})",
        "3pt": f"{avg_fg3m:.1f}/{avg_fg3a:.1f} ({format_pct(avg_fg3_pct)})",
        "ft": f"{avg_ftm:.1f}/{avg_fta:.1f} ({format_pct(avg_ft_pct)})",
    }


@st.cache_data(ttl=3600)
def get_all_players():
    return players.get_players()


def find_player_id(player_name: str):
    all_players = get_all_players()
    normalized_input = normalize_name(player_name)

    for player in all_players:
        if normalize_name(player["full_name"]) == normalized_input:
            return player["id"], player["full_name"]

    all_names = [normalize_name(player["full_name"]) for player in all_players]
    matches = get_close_matches(normalized_input, all_names, n=1, cutoff=0.6)

    if matches:
        matched_name = matches[0]
        for player in all_players:
            if normalize_name(player["full_name"]) == matched_name:
                return player["id"], player["full_name"]

    return None, None


def extract_opponent_abbr(matchup: str) -> str:
    if " vs. " in matchup:
        return matchup.split(" vs. ")[1].strip()

    if " @ " in matchup:
        return matchup.split(" @ ")[1].strip()

    return "N/A"


@st.cache_data(ttl=3600)
def get_nba_teams():
    return teams.get_teams()


def build_team_lookup_maps(standings_df: pd.DataFrame):
    nba_teams = get_nba_teams()
    team_id_to_abbr = {team["id"]: team["abbreviation"] for team in nba_teams}

    team_status_by_id = {}
    team_status_by_abbr = {}

    for _, row in standings_df.iterrows():
        team_id = safe_int(row["TeamID"])
        team_abbr = team_id_to_abbr.get(team_id, "N/A")

        record = str(row["Record"]) if "Record" in standings_df.columns else "N/A"
        conference = str(row["Conference"]) if "Conference" in standings_df.columns else "N/A"
        conf_rank = str(row["PlayoffRank"]) if "PlayoffRank" in standings_df.columns else "N/A"

        team_status = {
            "team_id": team_id,
            "abbr": team_abbr,
            "record": record,
            "conference": conference,
            "conf_rank": conf_rank,
        }

        team_status_by_id[team_id] = team_status
        if team_abbr != "N/A":
            team_status_by_abbr[team_abbr] = team_status

    return team_status_by_id, team_status_by_abbr


def get_game_score(game_id: str, team_id: int, score_cache: dict):
    if game_id in score_cache:
        return score_cache[game_id]

    try:
        sleep(SCORE_REQUEST_DELAY)

        data = retry_request(boxscore.BoxScore, game_id=game_id).get_dict()
        game = data.get("game", {})

        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})

        home_team_id = safe_int(home_team.get("teamId"), -1)
        away_team_id = safe_int(away_team.get("teamId"), -1)

        home_score = str(home_team.get("score", "N/A"))
        away_score = str(away_team.get("score", "N/A"))

        if team_id == home_team_id:
            result = (home_score, away_score)
        elif team_id == away_team_id:
            result = (away_score, home_score)
        else:
            result = ("N/A", "N/A")

        score_cache[game_id] = result
        return result

    except (KeyError, ValueError, TypeError, ReadTimeout, RequestException):
        result = ("N/A", "N/A")
        score_cache[game_id] = result
        return result


def build_missed_games_summary(player_games_df: pd.DataFrame, team_games_df: pd.DataFrame):
    player_game_ids = set(player_games_df["Game_ID"].astype(str).tolist())
    team_games_sorted = team_games_df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    total_missed_games = 0
    recent_missed_streak = 0
    absence_streaks = 0

    in_absence = False
    first_played_found = False

    for _, row in team_games_sorted.iterrows():
        game_id = str(row["Game_ID"])
        played = game_id in player_game_ids

        if not played:
            total_missed_games += 1
            if not in_absence:
                absence_streaks += 1
                in_absence = True
        else:
            in_absence = False

        if not first_played_found:
            if played:
                first_played_found = True
            else:
                recent_missed_streak += 1

    return {
        "total_missed_games": total_missed_games,
        "absence_streaks": absence_streaks,
        "recent_missed_streak": recent_missed_streak,
    }


def count_team_games_since_last_selected_game(selected_games_df: pd.DataFrame, team_games_df: pd.DataFrame) -> int:
    if selected_games_df.empty or team_games_df.empty:
        return 0

    last_selected_game_date = selected_games_df["GAME_DATE"].min()
    later_team_games = team_games_df[team_games_df["GAME_DATE"] > last_selected_game_date]

    return len(later_team_games) + 1


def parse_schedule_datetime(game: dict):
    possible_values = [
        game.get("gameDateTimeUTC"),
        game.get("gameDateEst"),
        game.get("gameDateTimeEst"),
        game.get("gameDate"),
    ]

    for value in possible_values:
        if not value:
            continue

        try:
            if isinstance(value, str):
                cleaned = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(cleaned)

                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return dt.astimezone()
        except ValueError:
            continue

    return None


def get_team_label(team_dict: dict):
    city = str(team_dict.get("teamCity", "")).strip()
    name = str(team_dict.get("teamName", "")).strip()
    tricode = str(team_dict.get("teamTricode", "")).strip()

    full_name = " ".join(part for part in [city, name] if part).strip()

    if full_name:
        return full_name
    if tricode:
        return tricode
    return "N/A"


def get_season_phase_label(game: dict):
    game_id = str(game.get("gameId", "")).strip()
    game_label = str(game.get("gameLabel", "")).strip().lower()
    series_text = str(game.get("seriesText", "")).strip().lower()
    game_subtype = str(game.get("gameSubtype", "")).strip().lower()
    bracket_text = str(game.get("bracketText", "")).strip().lower()

    combined = " ".join([game_label, series_text, game_subtype, bracket_text])

    if "play-in" in combined or "play in" in combined:
        return "Play-In"
    if "playoff" in combined or "playoffs" in combined:
        return "Playoffs"

    if len(game_id) >= 3:
        prefix = game_id[:3]
        if prefix == "005":
            return "Play-In"
        if prefix == "004":
            return "Playoffs"
        if prefix == "002":
            return "Regular Season"
        if prefix == "001":
            return "Preseason"

    stage_id = safe_int(game.get("seasonStageId"), -1)
    if stage_id == 2:
        return "Regular Season"
    if stage_id == 4:
        return "Playoffs"

    return "Regular Season"


@st.cache_data(ttl=1800)
def fetch_schedule_data():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    try:
        response = retry_request(
            requests.get,
            SCHEDULE_URL,
            headers=headers,
            timeout=STATS_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except (ReadTimeout, RequestException, ValueError):
        return {}


def get_next_team_game(team_id: int, team_abbr: str):
    schedule_data = fetch_schedule_data()

    if not schedule_data:
        return None

    league_schedule = schedule_data.get("leagueSchedule", {})
    game_dates = league_schedule.get("gameDates", [])
    now_local = datetime.now().astimezone()

    candidates = []

    for date_block in game_dates:
        games = date_block.get("games", [])

        for game in games:
            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})

            home_id = safe_int(home_team.get("teamId"), -1)
            away_id = safe_int(away_team.get("teamId"), -1)

            team_is_home = team_id == home_id
            team_is_away = team_id == away_id

            if not team_is_home and not team_is_away:
                home_tricode = str(home_team.get("teamTricode", "")).strip().upper()
                away_tricode = str(away_team.get("teamTricode", "")).strip().upper()

                if team_abbr.upper() == home_tricode:
                    team_is_home = True
                elif team_abbr.upper() == away_tricode:
                    team_is_away = True

            if not team_is_home and not team_is_away:
                continue

            game_dt = parse_schedule_datetime(game)
            if game_dt is None:
                continue

            if game_dt <= now_local:
                continue

            opponent_team = away_team if team_is_home else home_team
            venue_type = "Hazai" if team_is_home else "Idegenbeli"

            candidates.append(
                {
                    "datetime": game_dt,
                    "opponent": get_team_label(opponent_team),
                    "venue_type": venue_type,
                    "season_phase": get_season_phase_label(game),
                    "game_id": str(game.get("gameId", "N/A")),
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["datetime"])
    next_game = candidates[0]

    return {
        "date": next_game["datetime"].strftime("%Y-%m-%d"),
        "time": next_game["datetime"].strftime("%H:%M"),
        "datetime_label": next_game["datetime"].strftime("%Y-%m-%d %H:%M"),
        "opponent": next_game["opponent"],
        "venue_type": next_game["venue_type"],
        "season_phase": next_game["season_phase"],
        "game_id": next_game["game_id"],
    }


def standardize_games_df_dates(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return games_df.copy()

    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"],
        format="%b %d, %Y",
        errors="coerce",
    )
    df = df.dropna(subset=["GAME_DATE"])
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def add_min_decimal_column(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        df = games_df.copy()
        df["MIN_DECIMAL"] = pd.Series(dtype="float64")
        return df

    df = games_df.copy()

    if "MIN" not in df.columns:
        df["MIN"] = 0

    df["MIN_DECIMAL"] = df["MIN"].apply(parse_minutes_to_decimal)
    return df


def combine_game_logs(primary_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    if primary_df.empty and extra_df.empty:
        return pd.DataFrame()

    if primary_df.empty:
        combined = extra_df.copy()
    elif extra_df.empty:
        combined = primary_df.copy()
    else:
        all_columns = list(dict.fromkeys(list(primary_df.columns) + list(extra_df.columns)))
        combined = pd.concat(
            [
                primary_df.reindex(columns=all_columns),
                extra_df.reindex(columns=all_columns),
            ],
            ignore_index=True,
        )

    if "Game_ID" in combined.columns:
        combined["Game_ID"] = combined["Game_ID"].astype(str)
        combined = combined.drop_duplicates(subset=["Game_ID"], keep="first")

    combined = standardize_games_df_dates(combined)
    combined = add_min_decimal_column(combined)
    return combined


def fetch_playin_team_games(team_id: int, season: str) -> pd.DataFrame:
    try:
        df = retry_request(
            leaguegamefinder.LeagueGameFinder,
            player_or_team_abbreviation="T",
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="PlayIn",
            league_id_nullable="00",
            timeout=STATS_TIMEOUT,
        ).get_data_frames()[0]
    except (ReadTimeout, RequestException, ValueError, KeyError, TypeError, IndexError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    team_df = df.copy()
    if "GAME_ID" in team_df.columns and "Game_ID" not in team_df.columns:
        team_df["Game_ID"] = team_df["GAME_ID"].astype(str)
    elif "Game_ID" in team_df.columns:
        team_df["Game_ID"] = team_df["Game_ID"].astype(str)

    rename_map = {}
    if "TEAM_ABBREVIATION" in team_df.columns and "TEAM_ABBR" not in team_df.columns:
        rename_map["TEAM_ABBREVIATION"] = "TEAM_ABBR"
    if rename_map:
        team_df = team_df.rename(columns=rename_map)

    required_columns = [
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
    ]
    existing_columns = [col for col in required_columns if col in team_df.columns]
    team_df = team_df[existing_columns].copy()

    return standardize_games_df_dates(team_df)


def fetch_player_playin_games(player_id: int, team_playin_df: pd.DataFrame) -> pd.DataFrame:
    if team_playin_df.empty:
        return pd.DataFrame()

    player_rows = []

    for _, team_row in team_playin_df.iterrows():
        game_id = str(team_row["Game_ID"])

        try:
            sleep(PLAYIN_REQUEST_DELAY)

            box_df = retry_request(
                boxscoretraditionalv2.BoxScoreTraditionalV2,
                game_id=game_id,
                timeout=STATS_TIMEOUT,
            ).get_data_frames()[0]

            if box_df.empty:
                continue

            player_row_df = box_df[box_df["PLAYER_ID"].astype(int) == int(player_id)]
            if player_row_df.empty:
                continue

            row = player_row_df.iloc[0]

            oreb = safe_int(row.get("OREB", 0))
            dreb = safe_int(row.get("DREB", 0))
            reb = safe_int(row.get("REB", oreb + dreb))

            player_rows.append(
                {
                    "Game_ID": game_id,
                    "GAME_DATE": team_row.get("GAME_DATE"),
                    "MATCHUP": team_row.get("MATCHUP", "N/A"),
                    "WL": team_row.get("WL", "N/A"),
                    "MIN": row.get("MIN", "0"),
                    "FGM": safe_int(row.get("FGM", 0)),
                    "FGA": safe_int(row.get("FGA", 0)),
                    "FG3M": safe_int(row.get("FG3M", 0)),
                    "FG3A": safe_int(row.get("FG3A", 0)),
                    "FTM": safe_int(row.get("FTM", 0)),
                    "FTA": safe_int(row.get("FTA", 0)),
                    "OREB": oreb,
                    "DREB": dreb,
                    "REB": reb,
                    "AST": safe_int(row.get("AST", 0)),
                    "STL": safe_int(row.get("STL", 0)),
                    "BLK": safe_int(row.get("BLK", 0)),
                    "TOV": safe_int(row.get("TO", row.get("TOV", 0))),
                    "PTS": safe_int(row.get("PTS", 0)),
                }
            )
        except (ReadTimeout, RequestException, ValueError, KeyError, TypeError, IndexError):
            continue

    if not player_rows:
        return pd.DataFrame()

    df = pd.DataFrame(player_rows)
    df = standardize_games_df_dates(df)
    df = add_min_decimal_column(df)
    return df


def enrich_with_playin_if_needed(
    player_games_df: pd.DataFrame,
    team_games_df: pd.DataFrame,
    player_id: int,
    team_id: int,
    season: str,
    season_type: str,
):
    if season_type != "Playoffs":
        player_games_df = standardize_games_df_dates(player_games_df)
        player_games_df = add_min_decimal_column(player_games_df)
        team_games_df = standardize_games_df_dates(team_games_df)
        return player_games_df, team_games_df, False

    player_games_df = standardize_games_df_dates(player_games_df)
    player_games_df = add_min_decimal_column(player_games_df)
    team_games_df = standardize_games_df_dates(team_games_df)

    playin_team_df = fetch_playin_team_games(team_id=team_id, season=season)
    if playin_team_df.empty:
        return player_games_df, team_games_df, False

    playin_player_df = fetch_player_playin_games(
        player_id=player_id,
        team_playin_df=playin_team_df,
    )

    combined_player_df = combine_game_logs(player_games_df, playin_player_df)
    combined_team_df = combine_game_logs(team_games_df, playin_team_df)

    playin_added = not playin_team_df.empty
    return combined_player_df, combined_team_df, playin_added


def get_player_report(player_name: str, season: str, season_type: str, last_n_games: int) -> dict:
    if not player_name.strip():
        raise ValueError("Adj meg egy játékosnevet.")

    if last_n_games <= 0:
        raise ValueError("A meccsek száma csak pozitív egész lehet.")

    score_cache = {}

    player_id, real_player_name = find_player_id(player_name)

    if player_id is None:
        raise ValueError(f"Nincs találat erre a játékosra: {player_name}")

    player_info_df = retry_request(
        commonplayerinfo.CommonPlayerInfo,
        player_id=player_id,
        timeout=STATS_TIMEOUT,
    ).get_data_frames()[0]

    info_row = player_info_df.iloc[0]
    team_id = safe_int(info_row["TEAM_ID"])
    team_name = str(info_row["TEAM_NAME"])
    team_abbr = str(info_row["TEAM_ABBREVIATION"])

    player_games_df = retry_request(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
        timeout=STATS_TIMEOUT,
    ).get_data_frames()[0]

    if player_games_df.empty and season_type != "Playoffs":
        raise ValueError(f"Nincs meccsadat {real_player_name} számára a(z) {season} szezonban.")

    team_games_df = retry_request(
        teamgamelog.TeamGameLog,
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
        timeout=STATS_TIMEOUT,
    ).get_data_frames()[0]

    player_games_df, team_games_df, playin_added = enrich_with_playin_if_needed(
        player_games_df=player_games_df,
        team_games_df=team_games_df,
        player_id=player_id,
        team_id=team_id,
        season=season,
        season_type=season_type,
    )

    if player_games_df.empty:
        if season_type == "Playoffs":
            raise ValueError(
                f"{real_player_name} nem vett részt a {season} szezon rájátszás/play-in szakaszában."
            )
        raise ValueError("Nem található meccsadat.")

    standings_df = retry_request(
        leaguestandingsv3.LeagueStandingsV3,
        season=season,
        season_type=season_type,
        timeout=STATS_TIMEOUT,
    ).get_data_frames()[0]

    team_status_by_id, team_status_by_abbr = build_team_lookup_maps(standings_df)
    own_team_status = team_status_by_id.get(
        team_id,
        {
            "record": "N/A",
            "conference": "N/A",
            "conf_rank": "N/A",
        },
    )

    selected_games = player_games_df.head(last_n_games)
    actual_games_count = len(selected_games)

    if actual_games_count == 0:
        raise ValueError("Nem található meccsadat.")

    recent_averages = build_average_block(selected_games)
    season_averages = build_average_block(player_games_df)

    missed_summary = build_missed_games_summary(player_games_df, team_games_df)
    team_games_since_last_selected = count_team_games_since_last_selected_game(
        selected_games,
        team_games_df,
    )

    next_game_info = get_next_team_game(team_id=team_id, team_abbr=team_abbr)

    games_list = []

    for _, row in selected_games.iterrows():
        game_date = row["GAME_DATE"].strftime("%Y-%m-%d")
        opponent_abbr = extract_opponent_abbr(str(row["MATCHUP"]))
        opponent_status = team_status_by_abbr.get(
            opponent_abbr,
            {
                "record": "N/A",
                "conference": "N/A",
                "conf_rank": "N/A",
            },
        )

        own_score, opp_score = get_game_score(str(row["Game_ID"]), team_id, score_cache)

        fgm = safe_int(row["FGM"])
        fga = safe_int(row["FGA"])
        ftm = safe_int(row["FTM"])
        fta = safe_int(row["FTA"])
        fg3m = safe_int(row["FG3M"])
        fg3a = safe_int(row["FG3A"])

        fg_pct = pct_from_made_attempted(fgm, fga)
        ft_pct = pct_from_made_attempted(ftm, fta)
        fg3_pct = pct_from_made_attempted(fg3m, fg3a)

        games_list.append(
            {
                "Dátum": game_date,
                "Párosítás": str(row["MATCHUP"]),
                "Ellenfél konf.": opponent_status["conference"],
                "Ellenfél helyezés": opponent_status["conf_rank"],
                "Ellenfél mérleg": opponent_status["record"],
                "Eredmény": str(row["WL"]),
                "Állás": f"{own_score}-{opp_score}",
                "MIN": format_minutes_display(row.get("MIN", 0)),
                "FG": f"{fgm}/{fga} ({format_pct(fg_pct)})",
                "3PT": f"{fg3m}/{fg3a} ({format_pct(fg3_pct)})",
                "FT": f"{ftm}/{fta} ({format_pct(ft_pct)})",
                "PTS": safe_int(row["PTS"]),
                "REB": safe_int(row["REB"]),
                "AST": safe_int(row["AST"]),
                "STL": safe_int(row["STL"]),
                "BLK": safe_int(row["BLK"]),
                "TOV": safe_int(row["TOV"]),
            }
        )

    stats_section_label = "Alapszakasz"
    if season_type == "Playoffs":
        stats_section_label = "Rájátszás + Play-In" if playin_added else "Rájátszás"

    return {
        "player_name": real_player_name,
        "team_name": team_name,
        "team_abbr": team_abbr,
        "season": season,
        "season_type": season_type,
        "stats_section_label": stats_section_label,
        "playin_added": playin_added,
        "conference": own_team_status["conference"],
        "conf_rank": own_team_status["conf_rank"],
        "record": own_team_status["record"],
        "actual_games_count": actual_games_count,
        "season_games_count": len(player_games_df),
        "recent_averages": recent_averages,
        "season_averages": season_averages,
        "missed_games": missed_summary,
        "team_games_since_last_selected": team_games_since_last_selected,
        "next_game": next_game_info,
        "games": games_list,
    }


def render_average_section(title: str, averages: dict):
    st.markdown(title)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Perc", averages["min"])
    c2.metric("Pont", averages["pts"])
    c3.metric("Lepattanó", averages["reb"])
    c4.metric("Assziszt", averages["ast"])
    c5.metric("Labdaszerzés", averages["stl"])

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Blokk", averages["blk"])
    c7.metric("Eladott labda", averages["tov"])
    c8.metric("FG", averages["fg"])
    c9.metric("3PT", averages["3pt"])

    st.metric("FT", averages["ft"])


def main():
    st.set_page_config(page_title="NBA játékos statisztika", layout="wide")

    st.title("NBA játékos statisztika")
    st.write(
        "Egy játékos utolsó játszott meccseinek átlagai, teljes szezonos átlaga, "
        "következő csapatmeccse és meccsenkénti bontása."
    )

    with st.sidebar:
        st.header("Beállítások")
        player_name = st.text_input("Játékos neve", placeholder="pl. Nikola Jokic")
        season = st.text_input("Szezon", value=get_current_nba_season())
        season_type_label = st.selectbox("Szakasz", ["Alapszakasz", "Rájátszás"])
        last_n_games = st.number_input(
            "Hány utolsó játszott meccs?",
            min_value=1,
            max_value=82,
            value=5,
            step=1,
        )
        run_query = st.button("Lekérdezés", type="primary")

    st.info(
        "Megjegyzés: a kihagyott csapatmeccsek száma nem bizonyítja, "
        "hogy a kihagyás sérülés miatt történt."
    )

    st.markdown("### NBA fogadási oddsok")
    st.link_button(
        "Megnyitás: Vegas.hu NBA piacok",
        "https://vegas.hu/sports/kosarlabda/usa/nba",
        use_container_width=False,
    )

    if run_query:
        season_type = "Playoffs" if season_type_label == "Rájátszás" else "Regular Season"

        try:
            with st.spinner("Adatok lekérése..."):
                report = get_player_report(
                    player_name=player_name,
                    season=season.strip(),
                    season_type=season_type,
                    last_n_games=int(last_n_games),
                )

            st.subheader(report["player_name"])
            st.write(
                f"**Csapat:** {report['team_name']} ({report['team_abbr']})  \n"
                f"**Szezon:** {report['season']}  \n"
                f"**Lekért szakasz:** {report['stats_section_label']}  \n"
                f"**Konferencia:** {report['conference']}  \n"
                f"**Konferencia-helyezés:** {report['conf_rank']}  \n"
                f"**Mérleg:** {report['record']}"
            )

            if report["playin_added"]:
                st.success(
                    "A rájátszás nézethez a Play-In meccsek is hozzá lettek adva a statisztikákhoz."
                )

            st.markdown("## Következő csapatmeccs")
            next_game = report["next_game"]

            if next_game is None:
                st.warning("Nem sikerült következő meccset találni a menetrendben.")
            else:
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Dátum", next_game["date"])
                n2.metric("Idő", next_game["time"])
                n3.metric("Helyszín", next_game["venue_type"])
                n4.metric("Szakasz", next_game["season_phase"])

                st.write(
                    f"**Ellenfél:** {next_game['opponent']}  \n"
                    f"**Teljes időpont:** {next_game['datetime_label']}"
                )

            render_average_section(
                f"## Átlag az utolsó {report['actual_games_count']} játszott meccsen "
                f"({report['stats_section_label']})",
                report["recent_averages"],
            )

            render_average_section(
                f"## Teljes szakaszátlag ({report['season_games_count']} meccs, "
                f"{report['stats_section_label']})",
                report["season_averages"],
            )

            st.markdown("## Kihagyott csapatmeccsek")
            missed = report["missed_games"]

            c10, c11, c12 = st.columns(3)
            c10.metric("Kihagyott csapatmeccsek", missed["total_missed_games"])
            c11.metric("Hiányzási szakaszok", missed["absence_streaks"])
            c12.metric("Legutóbbi kihagyott sorozat", missed["recent_missed_streak"])

            st.write(
                f"**{report['player_name']}** legutóbbi "
                f"**{report['actual_games_count']}** játszott meccse óta a csapata "
                f"**{report['team_games_since_last_selected']}** meccset játszott."
            )

            st.markdown("## Meccsenkénti statisztikák")
            games_df = pd.DataFrame(report["games"])
            st.dataframe(games_df, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error(f"Hiba történt az adatok lekérése közben: {error}")


if __name__ == "__main__":
    main()
