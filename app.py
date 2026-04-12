import pandas as pd
import streamlit as st
from datetime import datetime
from difflib import get_close_matches
from time import sleep

from requests.exceptions import ReadTimeout, RequestException

from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguestandingsv3,
    playergamelog,
    teamgamelog,
)
from nba_api.stats.static import players, teams


STATS_TIMEOUT = 60
RETRY_COUNT = 3
RETRY_DELAY = 2
SCORE_REQUEST_DELAY = 0.6


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

    return len(later_team_games)


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

    if player_games_df.empty:
        if season_type == "Playoffs":
            raise ValueError(f"{real_player_name} nem vett részt a {season} szezon rájátszásában.")
        raise ValueError(f"Nincs meccsadat {real_player_name} számára a(z) {season} szezon alapszakaszában.")

    team_games_df = retry_request(
        teamgamelog.TeamGameLog,
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
        timeout=STATS_TIMEOUT,
    ).get_data_frames()[0]

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

    player_games_df["GAME_DATE"] = pd.to_datetime(
        player_games_df["GAME_DATE"],
        format="%b %d, %Y",
        errors="coerce",
    )
    player_games_df = player_games_df.dropna(subset=["GAME_DATE"])
    player_games_df = player_games_df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    team_games_df["GAME_DATE"] = pd.to_datetime(
        team_games_df["GAME_DATE"],
        format="%b %d, %Y",
        errors="coerce",
    )
    team_games_df = team_games_df.dropna(subset=["GAME_DATE"])
    team_games_df = team_games_df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    selected_games = player_games_df.head(last_n_games)
    actual_games_count = len(selected_games)

    if actual_games_count == 0:
        raise ValueError("Nem található meccsadat.")

    avg_pts = safe_float(selected_games["PTS"].mean())
    avg_reb = safe_float(selected_games["REB"].mean())
    avg_ast = safe_float(selected_games["AST"].mean())
    avg_stl = safe_float(selected_games["STL"].mean())
    avg_blk = safe_float(selected_games["BLK"].mean())
    avg_tov = safe_float(selected_games["TOV"].mean())

    avg_fgm = safe_float(selected_games["FGM"].mean())
    avg_fga = safe_float(selected_games["FGA"].mean())
    avg_ftm = safe_float(selected_games["FTM"].mean())
    avg_fta = safe_float(selected_games["FTA"].mean())
    avg_fg3m = safe_float(selected_games["FG3M"].mean())
    avg_fg3a = safe_float(selected_games["FG3A"].mean())

    total_fgm = safe_float(selected_games["FGM"].sum())
    total_fga = safe_float(selected_games["FGA"].sum())
    total_ftm = safe_float(selected_games["FTM"].sum())
    total_fta = safe_float(selected_games["FTA"].sum())
    total_fg3m = safe_float(selected_games["FG3M"].sum())
    total_fg3a = safe_float(selected_games["FG3A"].sum())

    avg_fg_pct = pct_from_made_attempted(total_fgm, total_fga)
    avg_ft_pct = pct_from_made_attempted(total_ftm, total_fta)
    avg_fg3_pct = pct_from_made_attempted(total_fg3m, total_fg3a)

    missed_summary = build_missed_games_summary(player_games_df, team_games_df)
    team_games_since_last_selected = count_team_games_since_last_selected_game(
        selected_games,
        team_games_df,
    )

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

    return {
        "player_name": real_player_name,
        "team_name": team_name,
        "team_abbr": team_abbr,
        "season": season,
        "season_type": season_type,
        "conference": own_team_status["conference"],
        "conf_rank": own_team_status["conf_rank"],
        "record": own_team_status["record"],
        "actual_games_count": actual_games_count,
        "averages": {
            "pts": round(avg_pts, 1),
            "reb": round(avg_reb, 1),
            "ast": round(avg_ast, 1),
            "stl": round(avg_stl, 1),
            "blk": round(avg_blk, 1),
            "tov": round(avg_tov, 1),
            "fg": f"{avg_fgm:.1f}/{avg_fga:.1f} ({format_pct(avg_fg_pct)})",
            "3pt": f"{avg_fg3m:.1f}/{avg_fg3a:.1f} ({format_pct(avg_fg3_pct)})",
            "ft": f"{avg_ftm:.1f}/{avg_fta:.1f} ({format_pct(avg_ft_pct)})",
        },
        "missed_games": missed_summary,
        "team_games_since_last_selected": team_games_since_last_selected,
        "games": games_list,
    }


def main():
    st.set_page_config(page_title="NBA játékos statisztika", layout="wide")

    st.title("NBA játékos statisztika")
    st.write("Egy játékos utolsó játszott meccseinek átlagai és meccsenkénti bontása.")

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
                f"**Szakasz:** {report['season_type']}  \n"
                f"**Konferencia:** {report['conference']}  \n"
                f"**Konferencia-helyezés:** {report['conf_rank']}  \n"
                f"**Mérleg:** {report['record']}"
            )

            st.markdown("## Átlag az utolsó játszott meccseken")
            avg = report["averages"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pont", avg["pts"])
            c2.metric("Lepattanó", avg["reb"])
            c3.metric("Assziszt", avg["ast"])
            c4.metric("Labdaszerzés", avg["stl"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Blokk", avg["blk"])
            c6.metric("Eladott labda", avg["tov"])
            c7.metric("FG", avg["fg"])
            c8.metric("3PT", avg["3pt"])

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("FT", avg["ft"])

            st.markdown("## Kihagyott csapatmeccsek")
            missed = report["missed_games"]

            c13, c14, c15 = st.columns(3)
            c13.metric("Kihagyott csapatmeccsek", missed["total_missed_games"])
            c14.metric("Hiányzási szakaszok", missed["absence_streaks"])
            c15.metric("Legutóbbi kihagyott sorozat", missed["recent_missed_streak"])

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
