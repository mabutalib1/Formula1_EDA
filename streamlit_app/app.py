import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

st.set_page_config(page_title="Formula 1, 1950–2024", page_icon="🏁", layout="wide")

RED = "#E10600"
INK = "#15151E"
GREY = "#8A8D93"
LIGHT = "#D9DBDF"
FONT = "Titillium Web, sans-serif"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input { font-family: 'Titillium Web', sans-serif; }
h1 { font-weight: 700; letter-spacing: -0.5px; border-left: 6px solid #E10600; padding-left: 14px; }
h2, h3 { font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 700; }
[data-testid="stMetricLabel"] { opacity: 0.7; }
.block-container { padding-top: 2.2rem; max-width: 1250px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv", parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except Exception:
        return None


def style(fig, height=380, x=None, y=None):
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=10, r=10, t=50, b=10), title_font=dict(size=17), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, title=x)
    fig.update_yaxes(gridcolor="#EEEFF1", title=y)
    return fig


def note(text):
    st.markdown(f"<p style='opacity:0.8; margin-top:-6px'>{text}</p>", unsafe_allow_html=True)


df = load_data()
model = load_model()

st.logo("logo.svg", size="large")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About the data")
    st.write(
        "Every Formula 1 World Championship race from 1950 to 2024: 1,125 races and about 25,000 "
        "driver results, with grid positions, finishing positions, teams, circuits and how each race ended."
    )
    st.caption("Source: Ergast F1 database (RaceOptiData), also available on Kaggle.")

    st.header("Filters")
    years = st.slider("Seasons", 1950, 2024, (1950, 2024))

    teams = df["team"].value_counts().index.tolist()
    team = st.selectbox("Team", ["All teams"] + teams)

    nationalities = df["nationality"].value_counts().index.tolist()
    nats = st.multiselect("Driver nationality", nationalities, placeholder="All nationalities")

# ---------------- Filtering ----------------
f = df[df["year"].between(years[0], years[1])]
if team != "All teams":
    f = f[f["team"] == team]
if nats:
    f = f[f["nationality"].isin(nats)]

# ---------------- Header ----------------
st.title("Formula 1, 1950–2024")
scope = f"{years[0]}–{years[1]}"
if team != "All teams":
    scope += f", {team}"
if nats:
    scope += f", {', '.join(nats)} drivers"
st.write(f"Seventy-five years of Grand Prix racing. Showing {scope}.")

if f.empty:
    st.warning("No results match these filters. Widen the season range or clear the nationality filter.")
    st.stop()

wins = f[f["win"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Races", f"{f['date'].nunique():,}")
c2.metric("Drivers", f"{f['driver'].nunique():,}")
c3.metric("Race winners", f"{wins['driver'].nunique():,}")
c4.metric("Finish rate", f"{f['finished'].mean():.0%}")

tabs = st.tabs(["Overview", "Drivers & teams", "Starting grid", "Circuits", "Driver lookup",
                "Podium predictor", "Data", "Key findings"])

# ---------------- Overview ----------------
with tabs[0]:
    left, right = st.columns(2)

    per_year = f.groupby("year")["date"].nunique().reset_index(name="races")
    fig = px.bar(per_year, x="year", y="races", title="Races per season", color_discrete_sequence=[INK])
    left.plotly_chart(style(fig), width="stretch", theme=None)

    finish = (f.groupby("year")["finished"].mean() * 100).reset_index(name="finished")
    fig = px.line(finish, x="year", y="finished", title="Share of cars reaching the finish (%)",
                  color_discrete_sequence=[RED])
    fig.update_yaxes(range=[0, 100])
    right.plotly_chart(style(fig), width="stretch", theme=None)

    mix = (pd.crosstab(f["decade"], f["result"], normalize="index") * 100).round(1)
    order = [c for c in ["Finished", "Mechanical", "Accident", "Other"] if c in mix.columns]
    mix = mix[order].reset_index().melt(id_vars="decade", var_name="result", value_name="percent")
    fig = px.bar(mix, x="decade", y="percent", color="result", title="How races ended, by decade (%)",
                 color_discrete_map={"Finished": "#3E8E5E", "Mechanical": INK, "Accident": RED, "Other": LIGHT})
    fig = style(fig, 400)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, title=None))
    st.plotly_chart(fig, width="stretch", theme=None)

    worst = finish.loc[finish["finished"].idxmin()]
    note(f"The lowest finish rate in this selection was {worst['finished']:.0f}% in {int(worst['year'])}. "
         f"Across all of F1, about half the cars finished until the late 1990s, and close to 90% do today. "
         f"Mechanical failure, not crashing, was the main reason cars retired.")

# ---------------- Drivers & teams ----------------
with tabs[1]:
    if wins.empty:
        st.info("No race wins in this selection.")
    else:
        left, right = st.columns(2)
        top_d = wins["driver"].value_counts().head(10).sort_values().reset_index()
        top_d.columns = ["driver", "wins"]
        fig = px.bar(top_d, x="wins", y="driver", orientation="h", title="Most race wins, drivers",
                     text="wins", color_discrete_sequence=[RED])
        left.plotly_chart(style(fig, 420), width="stretch", theme=None)

        top_t = wins["team"].value_counts().head(10).sort_values().reset_index()
        top_t.columns = ["team", "wins"]
        fig = px.bar(top_t, x="wins", y="team", orientation="h", title="Most race wins, teams",
                     text="wins", color_discrete_sequence=[INK])
        right.plotly_chart(style(fig, 420), width="stretch", theme=None)

        leader = top_d.iloc[-1]
        note(f"{leader['driver']} leads this selection with {leader['wins']} wins. "
             f"Only {wins['driver'].nunique()} of {f['driver'].nunique()} drivers here have won a race.")

        left, right = st.columns(2)
        fig = px.box(wins, x="decade", y="age", title="Age of race winners by decade",
                     color_discrete_sequence=[RED], category_orders={"decade": sorted(wins["decade"].unique())})
        left.plotly_chart(style(fig), width="stretch", theme=None)

        nat = f.drop_duplicates("driver")["nationality"].value_counts().head(10).sort_values().reset_index()
        nat.columns = ["nationality", "drivers"]
        fig = px.bar(nat, x="drivers", y="nationality", orientation="h", title="Drivers by nationality",
                     color_discrete_sequence=[GREY])
        right.plotly_chart(style(fig), width="stretch", theme=None)

        ages = wins.groupby("decade")["age"].median()
        note(f"The median race winner was {ages.iloc[0]:.0f} in the {ages.index[0]} and "
             f"{ages.iloc[-1]:.0f} in the {ages.index[-1]}.")

# ---------------- Starting grid ----------------
with tabs[2]:
    left, right = st.columns(2)

    g = f[f["grid"].between(1, 20)].groupby("grid")["win"].mean().mul(100).reset_index(name="win_pct")
    g["colour"] = [RED if x == 1 else LIGHT for x in g["grid"]]
    fig = go.Figure(go.Bar(x=g["grid"], y=g["win_pct"], marker_color=g["colour"],
                           hovertemplate="Grid %{x}: %{y:.1f}% wins<extra></extra>"))
    fig.update_layout(title="Chance of winning by starting position (%)")
    fig.update_xaxes(dtick=1)
    left.plotly_chart(style(fig, x="Grid position"), width="stretch", theme=None)

    if not wins.empty:
        pole = wins.groupby("decade")["pole"].mean().mul(100).reset_index(name="pct")
        fig = px.bar(pole, x="decade", y="pct", title="Races won from pole position, by decade (%)",
                     color_discrete_sequence=[INK])
        right.plotly_chart(style(fig), width="stretch", theme=None)

    if not g.empty and 1 in g["grid"].values:
        note(f"Starting from pole wins {g.loc[g['grid'] == 1, 'win_pct'].iloc[0]:.0f}% of the time in this selection. "
             "Since 2000, about half of all races have been won from pole.")

    if not wins.empty:
        share = wins.groupby(["year", "team"]).size().reset_index(name="wins")
        share["pct"] = share["wins"] / share["year"].map(wins.groupby("year").size()) * 100
        best = share.sort_values("pct", ascending=False).drop_duplicates("year").sort_values("year")
        fig = px.bar(best, x="year", y="pct", hover_data=["team", "wins"], color_discrete_sequence=[RED],
                     title="Share of the season's wins taken by the strongest team (%)")
        st.plotly_chart(style(fig, 360), width="stretch", theme=None)
        top = best.sort_values("pct", ascending=False).iloc[0]
        note(f"The most dominant season here: {top['team']} in {int(top['year'])}, winning {top['pct']:.0f}% of races.")

# ---------------- Circuits ----------------
with tabs[3]:
    circ = f.drop_duplicates("date").groupby(["circuit", "country", "lat", "lng"]).size().reset_index(name="races")
    fig = px.scatter_geo(circ, lat="lat", lon="lng", size="races", hover_name="circuit",
                         hover_data={"country": True, "races": True, "lat": False, "lng": False},
                         color_discrete_sequence=[RED], size_max=28, projection="natural earth")
    fig.update_geos(showcountries=True, countrycolor="#FFFFFF", landcolor="#ECEDEF",
                    showocean=True, oceancolor="#FFFFFF", showframe=False, coastlinecolor="#FFFFFF")
    fig.update_layout(title="Where the races were held (bubble size = races)")
    st.plotly_chart(style(fig, 480), width="stretch", theme=None)

    left, right = st.columns([2, 3])
    left.dataframe(circ.sort_values("races", ascending=False)[["circuit", "country", "races"]].head(10),
                   hide_index=True, width="stretch")
    with right:
        note(f"{len(circ)} circuits in this selection. Monza and Monaco have hosted the most races in F1 history, "
             "and about 60% of all Grands Prix have been held in Europe.")

# ---------------- Driver lookup ----------------
with tabs[4]:
    drivers = f.groupby("driver").agg(wins=("win", "sum"), starts=("date", "size"))
    drivers = drivers.sort_values(["wins", "starts"], ascending=False).index.tolist()
    name = st.selectbox("Choose a driver", drivers)
    d = f[f["driver"] == name]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Starts", len(d))
    c2.metric("Wins", int(d["win"].sum()))
    c3.metric("Podiums", int(d["podium"].sum()))
    c4.metric("Poles", int(d["pole"].sum()))
    c5.metric("Seasons", d["year"].nunique())
    st.caption(f"Raced from {d['year'].min()} to {d['year'].max()} for {d['team'].nunique()} team(s).")

    per = d.groupby("year").agg(wins=("win", "sum"), podiums=("podium", "sum")).reset_index()
    per["Other podiums"] = per["podiums"] - per["wins"]
    per = per.rename(columns={"wins": "Wins"})
    fig = px.bar(per, x="year", y=["Wins", "Other podiums"], title=f"{name}: wins and podiums per season",
                 color_discrete_map={"Wins": RED, "Other podiums": LIGHT})
    fig = style(fig, 340)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2, title=None))
    fig.update_xaxes(dtick=1 if per["year"].nunique() < 15 else None)
    st.plotly_chart(fig, width="stretch", theme=None)

    teams_d = d.groupby("team").agg(Seasons=("year", lambda s: f"{s.min()}–{s.max()}"), Starts=("date", "size"),
                                    Wins=("win", "sum")).sort_values("Starts", ascending=False)
    teams_d.index.name = "Team"
    st.dataframe(teams_d, width="stretch")

# ---------------- Podium predictor ----------------
with tabs[5]:
    st.subheader("What are the chances of a podium?")
    st.write("A random forest trained on every race since 1950 estimates the chance of a top-3 finish "
             "from the starting position, the season and the driver's age.")
    if model is None:
        st.info("The model file (model.joblib) was not found next to app.py.")
    else:
        left, right = st.columns([1, 2])
        with left:
            grid = st.number_input("Starting position", 1, 26, 3)
            year = st.slider("Season", 1950, 2024, 2024)
            age = st.slider("Driver age", 17, 50, 27)
            x = pd.DataFrame({"grid": [grid], "year": [year], "age": [age]})
            prob = model.predict_proba(x)[0, 1]
            st.metric("Podium chance", f"{prob:.0%}")
        with right:
            curve = pd.DataFrame({"grid": range(1, 21), "year": year, "age": age})
            curve["chance"] = model.predict_proba(curve[["grid", "year", "age"]])[:, 1] * 100
            curve["colour"] = [RED if x == grid else LIGHT for x in curve["grid"]]
            fig = go.Figure(go.Bar(x=curve["grid"], y=curve["chance"], marker_color=curve["colour"],
                                   hovertemplate="Grid %{x}: %{y:.0f}%<extra></extra>"))
            fig.update_layout(title=f"Podium chance by starting position ({year}, age {age})")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(style(fig, 360, x="Grid position"), width="stretch", theme=None)
        st.caption("The model does not know the car or the team, so it treats a Ferrari and a back-marker the same. "
                   "It is about 89% accurate on unseen races, against 87% for always guessing no podium.")

# ---------------- Data ----------------
with tabs[6]:
    st.subheader("Data preview")
    cols = ["year", "race", "circuit", "driver", "nationality", "team", "grid", "position", "points", "status"]
    st.dataframe(f[cols].sort_values(["year", "race", "position"]).head(1000), hide_index=True,
                 width="stretch")
    st.caption(f"First 1,000 of {len(f):,} results in this selection.")

    st.subheader("Summary statistics")
    st.dataframe(f[["grid", "position", "points", "laps", "age"]].describe().round(1).T, width="stretch")

    st.download_button("Download this selection (CSV)", f.to_csv(index=False), "f1_selection.csv", "text/csv")

# ---------------- Key findings ----------------
with tabs[7]:
    st.subheader("Key findings, 1950–2024")
    st.markdown("""
- **Reliability changed the sport.** Until the late 1990s about half the cars failed to finish, mostly because of
  mechanical problems. In 2024, 89% of starters reached the flag.
- **Qualifying decides a lot.** The pole-sitter wins 42% of races. Since 2000, about half of all races have been won from pole,
  up from 28% in the 1980s.
- **Success is concentrated.** Only 115 of 789 drivers have won a race. Hamilton (105), Schumacher (91) and Verstappen (63)
  lead the drivers; Ferrari (249 wins) leads the teams.
- **Single teams can own a season.** Red Bull won 95% of the races in 2023, McLaren 94% in 1988 and Mercedes 90% in 2016.
- **Winners are younger.** The median winner was 34 in the 1950s and 26 in the 2020s.
- **F1 grew and spread.** From 7 races in 1950 to 24 in 2024; Europe still hosts about 60% of all Grands Prix ever run.
""")
    st.caption("Limitations: rules, points systems and field sizes changed many times, so eras are not directly comparable. "
               "The Indianapolis 500 counted towards the championship from 1950 to 1960.")
